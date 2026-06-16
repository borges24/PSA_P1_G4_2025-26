import cv2
import numpy as np
import os
import sys
import glob
import ctypes
from ctypes import *
from rembg import remove
from retificacao import Retificador

# =============================================================================
# SDK Hikrobot
# =============================================================================
SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
if SDK_PATH not in sys.path:
    sys.path.append(SDK_PATH)

try:
    from MvCameraControl_class import *
    from CameraParams_header import *
except ImportError:
    print("[ERRO] Não foi possível importar o SDK da Hikrobot.")
    sys.exit(1)

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
pasta_base = rf"C:\Users\j0a0p\Desktop\PSA_G4_AV\Fotos_Asa_teste"

CAMERA_IP = "192.168.2.30"

# --- Parâmetros da câmara: DEVEM ser iguais aos do C2R (reconhecimento) ---
# para que as fotos-modelo tenham a mesma luz que terás em produção.
CAMERA_EXPOSURE_US = 30000.0   # microsegundos (subido de 15000; sobe mais se escuro)
CAMERA_GAIN        = 10.0      # dB (subido de 5; sobe com cuidado: ganho alto = ruido)
CAMERA_FRAME_RATE  = 15.0      # fps

# --- Validação do recorte do rembg ---
AREA_MINIMA_RECORTE   = 5000   # px²; recorte abaixo disto é considerado falha
RACIO_FRAGMENTO_MAX   = 0.05   # 2º maior contorno não pode passar 5% do principal

# --- Interruptor da retificacao geometrica ---
# DEVE ter o MESMO valor que o C2R. Se o C2R reconhece sem retificacao,
# os modelos tambem tem de ser tirados sem retificacao (e vice-versa),
# senao comparas formas processadas de maneira diferente.
USAR_RETIFICACAO = False

# APAGÁMOS AS VARIÁVEIS LARGURA_CAPTURA E ALTURA_CAPTURA DAQUI!

# Função auxiliar para gerir as pastas e o contador dinamicamente
def atualizar_pasta_e_contador(id_atual, estado_face):
    nome_pasta = f"Asa_{id_atual}_{estado_face}"
    caminho = os.path.join(pasta_base, nome_pasta)
    
    if not os.path.exists(caminho):
        os.makedirs(caminho)
        print(f"[INFO] Nova pasta criada: {caminho}")
    
    fotos_existentes = glob.glob(os.path.join(caminho, "*.png"))
    proximo_contador = len(fotos_existentes)
    return caminho, proximo_contador

# =============================================================================
# CLASSE DA CÂMARA
# =============================================================================
class CameraHikrobot:
    def __init__(self):
        self.cam = MvCamera()
        self.stOutFrame = MV_FRAME_OUT()
        memset(byref(self.stOutFrame), 0, sizeof(self.stOutFrame))
        self.buf_convert     = None
        self.buf_convert_len = 0

    def _set_exposicao(self):
        try:
            if CAMERA_EXPOSURE_US is not None:
                self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
                self.cam.MV_CC_SetFloatValue("ExposureTime", float(CAMERA_EXPOSURE_US))
            if CAMERA_GAIN is not None:
                self.cam.MV_CC_SetEnumValue("GainAuto", 0)
                self.cam.MV_CC_SetFloatValue("Gain", float(CAMERA_GAIN))
            if CAMERA_FRAME_RATE is not None:
                self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(CAMERA_FRAME_RATE))
            print("[OK] Exposicao/ganho fixados (iguais ao C2R).")
        except Exception as e:
            print(f"[AVISO] Nao foi possivel fixar exposicao: {e}")

    def ligar(self):
        MvCamera.MV_CC_Initialize()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)
        
        cam_idx = None
        for i in range(deviceList.nDeviceNum):
            info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            ip = info.SpecialInfo.stGigEInfo.nCurrentIp
            ip_str = f"{(ip & 0xff000000) >> 24}.{(ip & 0x00ff0000) >> 16}.{(ip & 0x0000ff00) >> 8}.{ip & 0x000000ff}"
            if ip_str == CAMERA_IP:
                cam_idx = i
                break

        if cam_idx is None:
            print(f"[ERRO] A câmara com o IP {CAMERA_IP} não foi encontrada!")
            return False

        stDevInfo = cast(deviceList.pDeviceInfo[cam_idx], POINTER(MV_CC_DEVICE_INFO)).contents
        if self.cam.MV_CC_CreateHandle(stDevInfo) != 0: return False
        if self.cam.MV_CC_OpenDevice() != 0: return False

        nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
        if nPacketSize > 0:
            self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetEnumValue("AcquisitionMode", 2) 
        self._set_exposicao()
        
        return True

    def iniciar_captura(self):
        return self.cam.MV_CC_StartGrabbing() == 0

    def capturar_frame_direto(self):
        if self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1000) != 0:
            return None
        try:
            info = self.stOutFrame.stFrameInfo
            w, h = int(info.nWidth), int(info.nHeight)

            if info.enPixelType == PixelType_Gvsp_Mono8:
                raw = ctypes.string_at(self.stOutFrame.pBufAddr, w * h)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            # Conversão OFICIAL do SDK -> BGR8 (mesmo método do C2R)
            tam_dst = w * h * 3
            if self.buf_convert is None or self.buf_convert_len < tam_dst:
                self.buf_convert     = (c_ubyte * tam_dst)()
                self.buf_convert_len = tam_dst

            stConv = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConv), 0, sizeof(stConv))
            stConv.nWidth         = w
            stConv.nHeight        = h
            stConv.pSrcData       = self.stOutFrame.pBufAddr
            stConv.nSrcDataLen    = info.nFrameLen
            stConv.enSrcPixelType = info.enPixelType
            stConv.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            stConv.pDstBuffer     = self.buf_convert
            stConv.nDstBufferSize = tam_dst

            if self.cam.MV_CC_ConvertPixelType(stConv) != 0:
                return None

            raw = ctypes.string_at(self.buf_convert, tam_dst)
            return np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        except Exception:
            return None
        finally:
            self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)

    def parar(self):
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()

# =============================================================================
# VALIDAÇÃO DO RECORTE
# =============================================================================
def validar_recorte(resultado_bgra):
    """Verifica se o recorte do rembg tem um único objeto sólido.
    Devolve (ok, mensagem). Usa o mesmo canal alpha que o C2R vai ler."""
    if resultado_bgra.shape[2] != 4:
        return False, "Sem canal alpha."

    alpha = resultado_bgra[:, :, 3]
    _, mascara = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    res = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # findContours devolve 2 ou 3 valores conforme a versao do OpenCV.
    # Pegar sempre no penultimo elemento garante a lista de contornos.
    contornos = res[-2]

    if not contornos:
        return False, "Recorte vazio (nada detetado)."

    # sorted() aceita tuplo OU lista e devolve sempre lista (evita .sort em tuplo)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    area_principal = cv2.contourArea(contornos[0])

    if area_principal < AREA_MINIMA_RECORTE:
        return False, f"Objeto demasiado pequeno ({int(area_principal)} px2)."

    # Verifica fragmentos soltos (ruído que o C2R apanharia como contorno extra)
    if len(contornos) > 1:
        area_segundo = cv2.contourArea(contornos[1])
        if area_segundo > area_principal * RACIO_FRAGMENTO_MAX:
            return False, f"Recorte com fragmentos soltos ({len(contornos)} objetos)."

    # Verifica que toca pouco nos bordos (asa cortada pela margem do frame)
    h, w = mascara.shape
    bordo = (mascara[0, :].sum() + mascara[-1, :].sum() +
             mascara[:, 0].sum() + mascara[:, -1].sum()) / 255
    if bordo > (w + h) * 0.15:
        return False, "Asa cortada pela margem do frame."

    return True, f"OK ({int(area_principal)} px2, {len(contornos)} objeto/s)."

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    camara = CameraHikrobot()
    if not camara.ligar() or not camara.iniciar_captura():
        sys.exit(1)

    # Retificador geometrico. So ativo se USAR_RETIFICACAO=True (igual ao C2R).
    retificador = Retificador() if USAR_RETIFICACAO else None
    if retificador is None:
        print("[INFO] Retificacao DESLIGADA -> fotos na imagem original.")

    ID_ASA = 0
    ESTADO_FACE = "Normal" 
    caminho_guardar, contador = atualizar_pasta_e_contador(ID_ASA, ESTADO_FACE)

    print(f"\n[INFO] O sistema arrancou na Asa {ID_ASA} (Face: {ESTADO_FACE}).")
    print(" COMANDOS:")
    print(" -> Teclas '0' a '9': Muda a asa.")
    print(" -> Tecla 'R': Roda a asa.")
    print(" -> Pressiona 'S': Foto.")
    print(" -> Pressiona 'Q': Sair.")

    try:
        while True:
            frame = camara.capturar_frame_direto()
            if frame is None:
                continue

            # Corrige lente + perspetiva SE ativo. Senao, usa o frame original.
            if retificador is not None:
                frame = retificador.aplicar(frame)
            # =================================================================
            # CÁLCULO DINÂMICO DE PROPORÇÃO
            # Como a imagem que chega (frame) já vem com a resolução certa,
            # basta ler o shape (tamanho) dela e calcular a proporção.
            # =================================================================
            h_real, w_real = frame.shape[:2]
            proporcao = w_real / h_real
            
            display_width = 1024
            display_height = int(display_width / proporcao)
            
            frame_display = cv2.resize(frame, (display_width, display_height))
            # =================================================================
            
            cv2.rectangle(frame_display, (0, 0), (display_width, 60), (0, 0, 0), -1)
            texto_linha1 = f"ALVO: Asa {ID_ASA} | ESTADO: {ESTADO_FACE.upper()} | Fotos: {contador} | Res: {w_real}x{h_real}"
            texto_linha2 = "0-9: Mudar | R: Rodar | S: Foto | Q: Sair"
            
            cor_estado = (0, 255, 0) if ESTADO_FACE == "Normal" else (0, 100, 255)
            cv2.putText(frame_display, texto_linha1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_estado, 2)
            cv2.putText(frame_display, texto_linha2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            cv2.imshow('Captura Hikrobot', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if ord('0') <= key <= ord('9'):
                ID_ASA = key - ord('0')
                caminho_guardar, contador = atualizar_pasta_e_contador(ID_ASA, ESTADO_FACE)
                print(f"\n[INFO] Asa {ID_ASA} ({ESTADO_FACE}).")
            
            elif key == ord('r') or key == ord('R'):
                ESTADO_FACE = "Invertida" if ESTADO_FACE == "Normal" else "Normal"
                caminho_guardar, contador = atualizar_pasta_e_contador(ID_ASA, ESTADO_FACE)
                print(f"\n[INFO] Estado: {ESTADO_FACE.upper()}.")

            elif key == ord('s') or key == ord('S'):
                print(f"\n[A PROCESSAR] A remover fundo...")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = remove(frame_rgb)
                resultado_bgra = cv2.cvtColor(resultado, cv2.COLOR_RGBA2BGRA)

                # Validação automática do recorte
                ok, msg = validar_recorte(resultado_bgra)
                print(f"[VALIDACAO] {msg}")

                nome_arquivo = os.path.join(caminho_guardar, f"asa_{ID_ASA}_{ESTADO_FACE.lower()}_foto_{contador}.png")

                preview = cv2.resize(resultado_bgra, (display_width, display_height))
                cor_v = (0, 255, 0) if ok else (0, 0, 255)
                etiqueta = "RECORTE OK" if ok else "RECORTE COM PROBLEMA"
                cv2.putText(preview, etiqueta, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_v, 2)
                cv2.imshow('Preview sem fundo', preview)

                if ok:
                    print(" -> 'ENTER' para GUARDAR ou 'ESC' para DESCARTAR.")
                    tecla = cv2.waitKey(0) & 0xFF
                    if tecla == 13:
                        cv2.imwrite(nome_arquivo, resultado_bgra)
                        print(f"[SUCESSO] Guardado: {nome_arquivo}")
                        contador += 1
                    else:
                        print("[INFO] Descartada.")
                else:
                    # Recorte mau: exige confirmação forçada para evitar modelos ruins
                    print(" -> Recorte reprovado na validacao.")
                    print(" -> 'F' para FORCAR gravacao mesmo assim, qualquer outra para descartar.")
                    tecla = cv2.waitKey(0) & 0xFF
                    if tecla == ord('f') or tecla == ord('F'):
                        cv2.imwrite(nome_arquivo, resultado_bgra)
                        print(f"[AVISO] Guardado FORCADO: {nome_arquivo}")
                        contador += 1
                    else:
                        print("[INFO] Descartada.")

                cv2.destroyWindow('Preview sem fundo')
                        
            elif key == ord('q') or key == ord('Q'):
                break

    finally:
        camara.parar()
        cv2.destroyAllWindows()
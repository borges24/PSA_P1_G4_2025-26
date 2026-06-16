"""
=============================================================
  CALIBRACAO DA CAMARA - Projeto 3 PSA 2025/26
  -- Versao adaptada para camara Hikrobot + lente Fujinon --
  -- HF35HA-1S  (1:1.6 / 35 mm, C-mount, 2/3", baixa distorcao) --
=============================================================
  Diferencas em relacao a versao standard:

    1. MODELO DE DISTORCAO REDUZIDO
       A HF35HA-1S e uma lente prime telephoto com distorcao
       muito baixa (< 0.1% segundo a Fujinon). Ajustar k1,k2,k3
       e p1,p2 leva a sobreajuste -> coeficientes irrealistas e
       maior erro de reprojecao. Aqui usamos:
           CALIB_ZERO_TANGENT_DIST  (p1 = p2 = 0)
           CALIB_FIX_K3             (k3 = 0)
       Ficamos apenas com k1 e k2, que e o adequado para esta lente.

    2. PALPITE INICIAL DA MATRIZ DA CAMARA (intrinsic guess)
       Sabemos a distancia focal nominal (35 mm). Com isto e o
       tamanho do pixel do sensor estimamos fx, fy iniciais, o
       que ajuda o solver a convergir mesmo com poucas amostras.

    3. EXPOSICAO MAIS CURTA
       A f/1.6 (abertura maxima) entra muito mais luz do que com
       lentes comuns. Por defeito reduzimos o tempo de exposicao
       e o ganho para evitar saturacao do tabuleiro branco.

    4. SEM DECIMATION (resolucao total)
       O FOV e estreito (~14 graus em sensor 2/3"); manter a
       resolucao total ajuda a deteccao precisa dos cantos.
       Se a camara nao acompanhar, podes voltar a ativar.

    5. MAIS FOTOS RECOMENDADAS
       Lentes de FOV estreito precisam de mais posicoes diversas
       (inclinacoes acentuadas, cantos da imagem) -> MIN_FOTOS=20.

    6. VERIFICACAO DE FOCO
       A profundidade de campo a f/1.6 e curta. O script avalia
       a nitidez (variancia do Laplaciano) e avisa se a imagem
       estiver desfocada antes de aceitar a foto.

  COMO USAR:
    1. Antes de comecar, fecha um pouco o diafragma da lente (ex.
       f/4 ou f/5.6) para teres profundidade de campo razoavel
       e o tabuleiro nitido em varias distancias.
    2. Foca a lente manualmente para a distancia tipica de trabalho.
    3. Corre o script: python calibrar_camera_hikrobot_fujinon35.py
    4. Move o tabuleiro -> diferentes posicoes, distancias e
       inclinacoes (inclui os 4 cantos da imagem!).
    5. Pressiona ESPACO quando aparecer "DETETADO".
    6. Tira pelo menos 20 fotos.
    7. Pressiona 'C' para calibrar.

  RESULTADO:
    -> calibracao_camera_fujinon35.npz na pasta atual
=============================================================
"""
import cv2
import numpy as np
import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime

# SDK Hikrobot --------------------------------------------------
SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
if os.path.exists(SDK_PATH):
    sys.path.insert(0, SDK_PATH)

try:
    from MvCameraControl_class import *
    from MvErrorDefine_const import *
    from CameraParams_header import *
    _SDK_DISPONIVEL = True
except ImportError:
    _SDK_DISPONIVEL = False
    print("[AVISO] SDK Hikrobot nao encontrado.")
    print(f"        Verifica se os ficheiros estao em: {SDK_PATH}")
    print("        O script ira correr em MODO DEMO.\n")


# ──────────────────────────────────────────────────────────────────
#  CONFIGURACOES
# ──────────────────────────────────────────────────────────────────

INDICE_CAMERA       = 0
COLUNAS             = 7    # cantos internos na horizontal
LINHAS              = 7    # cantos internos na vertical
TAMANHO_QUADRADO    = 25   # mm
MIN_FOTOS           = 20   # +5 que a versao base (FOV estreito precisa mais)
FICHEIRO_CALIBRACAO = "calibracao_camera_fujinon35.npz"
PASTA_CALIB         = "fotos_calibracao_fujinon35"
TIMEOUT_FRAME_MS    = 1000
MAX_FALHAS_FRAME    = 30

# ---- Parametros especificos da lente HF35HA-1S ------------------
FOCAL_LENGTH_MM      = 35.0     # distancia focal nominal da lente
PIXEL_SIZE_UM        = 3.45     # pixel pitch tipico de sensores 2/3" Hikrobot
                                # (ajusta para a tua camara: ver datasheet)
USAR_INTRINSIC_GUESS = True     # palpite inicial de fx,fy a partir do focal length
USAR_DECIMATION      = False    # FOV estreito -> melhor usar resolucao total
LIMIAR_NITIDEZ       = 80.0     # variancia minima do Laplaciano (medida na imagem
                                # apos CLAHE -> insensivel ao brilho global)

# Exposicao -- ajusta aqui se a imagem ficar escura ou queimada.
# Por defeito usamos os mesmos valores da versao base; reduz se a lente
# estiver muito aberta (f/1.6 ou f/2) e a imagem ficar saturada.
EXPOSURE_TIME_US = 40000.0   # 40 ms
GAIN_DB          = 15.0

# Flags de calibracao: lente de baixa distorcao -> evitar sobreajuste
FLAGS_CALIB = (
    cv2.CALIB_ZERO_TANGENT_DIST   # p1 = p2 = 0  (sem distorcao tangencial)
    | cv2.CALIB_FIX_K3            # k3 = 0
)


# ──────────────────────────────────────────────────────────────────
#  FUNCOES AUXILIARES SDK (iguais a versao base)
# ──────────────────────────────────────────────────────────────────

def _to_hex(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return "0x" + hexStr


def _is_mono(pixel_type):
    if not _SDK_DISPONIVEL:
        return False
    return pixel_type in [
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
    ]


def _is_color(pixel_type):
    if not _SDK_DISPONIVEL:
        return False
    return pixel_type in [
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGR10,
        PixelType_Gvsp_BayerRG10,
        PixelType_Gvsp_BayerGB10,
        PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerGR12,
        PixelType_Gvsp_BayerRG12,
        PixelType_Gvsp_BayerGB12,
        PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_BayerGR10_Packed,
        PixelType_Gvsp_BayerRG10_Packed,
        PixelType_Gvsp_BayerGB10_Packed,
        PixelType_Gvsp_BayerBG10_Packed,
        PixelType_Gvsp_BayerGR12_Packed,
        PixelType_Gvsp_BayerRG12_Packed,
        PixelType_Gvsp_BayerGB12_Packed,
        PixelType_Gvsp_BayerBG12_Packed,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    ]


# ──────────────────────────────────────────────────────────────────
#  GESTOR DA CAMARA HIKROBOT
#  (seccao reconstruida a partir das classes de camara do projeto)
# ──────────────────────────────────────────────────────────────────

class CameraCalib:
    """Camara Hikrobot para calibracao. Devolve frames BGR (ou cinza->BGR
    se a camara for mono). Fixa exposicao/ganho para o tabuleiro nao queimar."""

    def __init__(self, indice=0):
        self.indice    = indice
        self.cam       = None
        self.frame_out = None
        self.buf_conv  = None
        self.buf_len   = 0
        self._falhas   = 0

    def abrir(self):
        if not _SDK_DISPONIVEL:
            print("[CAMARA] Modo DEMO (sem SDK). Sem captura real.")
            return False

        MvCamera.MV_CC_Initialize()
        lista = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, lista)
        if ret != 0 or lista.nDeviceNum == 0:
            print("[ERRO] Nenhuma camara encontrada.")
            return False

        if self.indice >= lista.nDeviceNum:
            print(f"[ERRO] Indice {self.indice} invalido "
                  f"({lista.nDeviceNum} camara(s) detetada(s)).")
            return False

        info = cast(lista.pDeviceInfo[self.indice],
                    POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        if self.cam.MV_CC_CreateHandle(info) != 0:
            print("[ERRO] CreateHandle falhou."); return False
        if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            print("[ERRO] OpenDevice falhou."); return False

        # Pacote otimo (so faz sentido em GigE)
        if info.nTLayerType == MV_GIGE_DEVICE:
            n = self.cam.MV_CC_GetOptimalPacketSize()
            if n > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", n)

        # Aquisicao continua, sem trigger
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)
        self.cam.MV_CC_SetEnumValue("AcquisitionMode", 2)

        # Exposicao/ganho fixos (evita saturacao do tabuleiro a f/1.6)
        try:
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            self.cam.MV_CC_SetFloatValue("ExposureTime", float(EXPOSURE_TIME_US))
            self.cam.MV_CC_SetEnumValue("GainAuto", 0)
            self.cam.MV_CC_SetFloatValue("Gain", float(GAIN_DB))
        except Exception as e:
            print(f"[AVISO] Nao foi possivel fixar exposicao/ganho: {e}")

        self.frame_out = MV_FRAME_OUT()
        memset(byref(self.frame_out), 0, sizeof(self.frame_out))

        if self.cam.MV_CC_StartGrabbing() != 0:
            print("[ERRO] StartGrabbing falhou."); return False

        print("[CAMARA] Aberta e a capturar.")
        return True

    def capturar(self):
        """Devolve um frame BGR (numpy) ou None."""
        if not _SDK_DISPONIVEL or self.cam is None:
            return None

        if self.cam.MV_CC_GetImageBuffer(self.frame_out, TIMEOUT_FRAME_MS) != 0:
            self._falhas += 1
            if self._falhas >= MAX_FALHAS_FRAME:
                print(f"[AVISO] {self._falhas} falhas seguidas de captura.")
                self._falhas = 0
            return None

        self._falhas = 0
        try:
            info = self.frame_out.stFrameInfo
            w, h = int(info.nWidth), int(info.nHeight)
            pt   = info.enPixelType

            if pt == PixelType_Gvsp_Mono8:
                raw = ctypes.string_at(self.frame_out.pBufAddr, w * h)
                arr = np.frombuffer(raw, np.uint8).reshape((h, w))
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            # Converte qualquer formato (Bayer/YUV/...) para BGR via SDK
            tam = w * h * 3
            if self.buf_conv is None or self.buf_len < tam:
                self.buf_conv = (c_ubyte * tam)()
                self.buf_len  = tam

            conv = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(conv), 0, sizeof(conv))
            conv.nWidth         = w
            conv.nHeight        = h
            conv.pSrcData       = self.frame_out.pBufAddr
            conv.nSrcDataLen    = info.nFrameLen
            conv.enSrcPixelType = pt
            conv.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            conv.pDstBuffer     = self.buf_conv
            conv.nDstBufferSize = tam

            if self.cam.MV_CC_ConvertPixelType(conv) != 0:
                return None

            raw = ctypes.string_at(self.buf_conv, tam)
            return np.frombuffer(raw, np.uint8).reshape((h, w, 3)).copy()
        except Exception:
            return None
        finally:
            self.cam.MV_CC_FreeImageBuffer(self.frame_out)

    def fechar(self):
        if _SDK_DISPONIVEL and self.cam is not None:
            try:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass
        print("[CAMARA] Fechada.")


# ──────────────────────────────────────────────────────────────────
#  AUXILIARES DE IMAGEM / CALIBRACAO
# ──────────────────────────────────────────────────────────────────

def _aplicar_clahe(gray):
    """Equalizacao adaptativa -> deteccao de cantos robusta a iluminacao."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _nitidez(gray):
    """Variancia do Laplaciano: valor alto = imagem nitida."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimar_matriz_inicial(img_size):
    """Palpite inicial da matriz da camara a partir do focal length nominal
    e do tamanho do pixel. img_size = (largura, altura)."""
    w, h = img_size
    fx = (FOCAL_LENGTH_MM * 1000.0) / PIXEL_SIZE_UM   # mm->um / (um/px) = px
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def _criar_objp():
    """Pontos 3D do tabuleiro (z=0), em mm."""
    objp = np.zeros((LINHAS * COLUNAS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLUNAS, 0:LINHAS].T.reshape(-1, 2)
    objp *= TAMANHO_QUADRADO
    return objp


# ──────────────────────────────────────────────────────────────────
#  ROTINA PRINCIPAL DE CALIBRACAO
# ──────────────────────────────────────────────────────────────────

def calibrar():
    print("=" * 60)
    print("  CALIBRACAO DA CAMARA - Fujinon HF35HA-1S 35mm")
    print("=" * 60)
    print(f"  Tabuleiro: {COLUNAS}x{LINHAS} cantos internos, "
          f"quadrado {TAMANHO_QUADRADO} mm")
    print(f"  Minimo de fotos: {MIN_FOTOS}")
    print(f"  Resultado -> {FICHEIRO_CALIBRACAO}")
    print("=" * 60)
    print("  Move o tabuleiro por toda a imagem (inclui os 4 cantos),")
    print("  varia distancias e inclinacoes. ESPACO para guardar foto,")
    print("  'C' para calibrar, 'Q' para sair.\n")

    os.makedirs(PASTA_CALIB, exist_ok=True)

    cam = CameraCalib(INDICE_CAMERA)
    if not cam.abrir():
        print("[ERRO] Nao foi possivel abrir a camara. A terminar.")
        return

    # Criterio de refinamento sub-pixel dos cantos
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Pontos 3D do tabuleiro (mesmos para todas as fotos)
    objp = _criar_objp()

    objpoints = []   # 3D no mundo real
    imgpoints = []   # 2D na imagem
    n_fotos   = 0
    img_size  = None

    # Flags de deteccao do tabuleiro
    flags_find = (cv2.CALIB_CB_ADAPTIVE_THRESH
                  | cv2.CALIB_CB_NORMALIZE_IMAGE
                  | cv2.CALIB_CB_FAST_CHECK)

    try:
        while True:
            frame = cam.capturar()
            if frame is None:
                # em modo demo ou falha, evita loop apertado
                if not _SDK_DISPONIVEL:
                    print("[DEMO] Sem camara real; nada para mostrar.")
                    break
                continue

            if img_size is None:
                img_size = (frame.shape[1], frame.shape[0])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = _aplicar_clahe(gray)

            # Tenta detetar o tabuleiro (primeiro na imagem equalizada)
            found, corners = cv2.findChessboardCorners(
                gray_eq, (COLUNAS, LINHAS), flags_find)
            estrategia = "CLAHE"
            if not found:
                found, corners = cv2.findChessboardCorners(
                    gray, (COLUNAS, LINHAS), flags_find)
                estrategia = "RAW"

            nitidez = _nitidez(gray_eq)
            nitido  = nitidez >= LIMIAR_NITIDEZ

            # Preparar imagem de display (reduzida)
            escala_disp = 1024.0 / frame.shape[1]
            w_disp = 1024
            h_disp = int(frame.shape[0] * escala_disp)
            display = cv2.resize(frame, (w_disp, h_disp))

            # Preview CLAHE no canto
            preview_aux = cv2.cvtColor(_aplicar_clahe(gray), cv2.COLOR_GRAY2BGR)
            mini_w = w_disp // 4
            mini_h = int(preview_aux.shape[0] * (mini_w / preview_aux.shape[1]))
            mini    = cv2.resize(preview_aux, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
            display[10:10 + mini_h, w_disp - mini_w - 10:w_disp - 10] = mini
            cv2.rectangle(display,
                          (w_disp - mini_w - 10, 10),
                          (w_disp - 10, 10 + mini_h),
                          (180, 180, 180), 1)
            cv2.putText(display, "CLAHE preview",
                        (w_disp - mini_w - 5, 10 + mini_h + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, (COLUNAS, LINHAS), corners2, found)

                if nitido:
                    cv2.putText(display,
                                f"DETETADO ({estrategia}) - NITIDO - ESPACO p/ foto",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display,
                                f"DETETADO ({estrategia}) MAS DESFOCADO - ajusta foco",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                corners2 = None
                cv2.putText(display,
                            "Tabuleiro nao detetado - aproxima/afasta ou melhora luz",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display,
                            "Dica: evita brilhos/sombras + verifica foco da lente",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.putText(display, f"Nitidez: {nitidez:.0f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if nitido else (0, 165, 255), 1)

            cor_cont = (0, 255, 0) if n_fotos >= MIN_FOTOS else (0, 165, 255)
            cv2.putText(display, f"Fotos: {n_fotos}/{MIN_FOTOS}",
                        (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor_cont, 2)

            if n_fotos >= MIN_FOTOS:
                cv2.putText(display, "Prima 'C' para calibrar",
                            (display.shape[1] - 280, display.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow("Calibracao Fujinon 35mm [Hikrobot]", display)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord(' ') and found and corners2 is not None and nitido:
                objpoints.append(objp)
                imgpoints.append(corners2)
                n_fotos += 1
                ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
                caminho = os.path.join(PASTA_CALIB, f"calib_{n_fotos:02d}_{ts}.jpg")
                cv2.imwrite(caminho, frame)
                print(f"[FOTO {n_fotos:02d}] Guardada ({estrategia}, nitidez {nitidez:.0f}) -> {caminho}")

            elif tecla == ord(' ') and found and not nitido:
                print(f"[AVISO] Foto rejeitada por desfoque (nitidez {nitidez:.0f} < {LIMIAR_NITIDEZ:.0f}). Ajusta o foco.")

            elif tecla in (ord('c'), ord('C')) and n_fotos >= MIN_FOTOS:
                print(f"\n[CALIBRAR] A calcular com {n_fotos} fotos...")
                print(f"[CALIBRAR] Modelo: k1, k2 fixados (k3=0, p1=p2=0).")

                # Construir matriz inicial se pedido
                K_init    = None
                dist_init = None
                flags     = FLAGS_CALIB
                if USAR_INTRINSIC_GUESS:
                    K_init    = estimar_matriz_inicial(img_size)
                    dist_init = np.zeros((5, 1), dtype=np.float64)
                    flags    |= cv2.CALIB_USE_INTRINSIC_GUESS
                    print(f"[CALIBRAR] Palpite inicial -> fx={K_init[0,0]:.0f}, fy={K_init[1,1]:.0f}")

                ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, img_size, K_init, dist_init, flags=flags
                )

                if ret_cal:
                    erro_total = 0.0
                    for i in range(len(objpoints)):
                        pts2d, _ = cv2.projectPoints(
                            objpoints[i], rvecs[i], tvecs[i], mtx, dist
                        )
                        erro_total += cv2.norm(imgpoints[i], pts2d, cv2.NORM_L2) / len(pts2d)
                    erro_medio = erro_total / len(objpoints)

                    # Distancia focal recuperada em mm (sanity check)
                    fx_mm = mtx[0, 0] * (PIXEL_SIZE_UM / 1000.0)
                    fy_mm = mtx[1, 1] * (PIXEL_SIZE_UM / 1000.0)

                    print(f"\n[OK] Calibracao concluida!")
                    print(f"     Erro medio de reprojecao : {erro_medio:.4f} px")
                    print(f"     (abaixo de 1.0 px e bom; abaixo de 0.5 px e excelente)")
                    print(f"\n     Matriz da camera:")
                    print(f"       fx = {mtx[0,0]:.2f} px ({fx_mm:.2f} mm)  fy = {mtx[1,1]:.2f} px ({fy_mm:.2f} mm)")
                    print(f"       cx = {mtx[0,2]:.2f}    cy = {mtx[1,2]:.2f}")
                    print(f"     Distorcao (k1, k2, p1, p2, k3):")
                    print(f"       {dist.ravel()}")
                    print(f"     (esperado para HF35HA-1S: |k1|, |k2| pequenos < 0.1)")
                    print(f"     Focal recuperada vs nominal (35 mm): diferenca {abs(fx_mm - FOCAL_LENGTH_MM):.2f} mm")

                    np.savez(
                        FICHEIRO_CALIBRACAO,
                        camera_matrix   = mtx,
                        dist_coeffs     = dist,
                        img_size        = np.array(list(img_size)),
                        erro_reprojecao = np.array([erro_medio]),
                        lente           = np.array(["Fujinon HF35HA-1S 35mm f/1.6"]),
                        focal_mm        = np.array([FOCAL_LENGTH_MM]),
                        pixel_size_um   = np.array([PIXEL_SIZE_UM]),
                    )
                    print(f"\n[GUARDADO] {FICHEIRO_CALIBRACAO}")
                    print("[FIM] Podes fechar e usar o sistema de visao.")

                    # Ecra de resultado visual
                    frame_final = cam.capturar()
                    if frame_final is not None:
                        resultado = frame_final.copy()
                    else:
                        resultado = np.zeros((600, 900, 3), dtype=np.uint8)

                    overlay = resultado.copy()
                    cv2.rectangle(overlay, (0, 0), (resultado.shape[1], resultado.shape[0]),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.55, resultado, 0.45, 0, resultado)

                    linhas_texto = [
                        ("CALIBRACAO CONCLUIDA COM SUCESSO!", (0, 255, 100), 1.1, 3),
                        (f"Lente: Fujinon HF35HA-1S  (35 mm, f/1.6)",
                         (255, 255, 255), 0.65, 1),
                        (f"Erro medio de reprojecao: {erro_medio:.4f} px",
                         (255, 255, 255), 0.75, 2),
                        (f"fx={mtx[0,0]:.0f}  fy={mtx[1,1]:.0f}  cx={mtx[0,2]:.0f}  cy={mtx[1,2]:.0f}",
                         (200, 200, 200), 0.65, 1),
                        (f"k1={dist.ravel()[0]:.5f}  k2={dist.ravel()[1]:.5f}",
                         (200, 200, 200), 0.60, 1),
                        (f"Ficheiro guardado: {FICHEIRO_CALIBRACAO}",
                         (0, 220, 255), 0.65, 2),
                        ("A fechar em 5 segundos...", (160, 160, 160), 0.60, 1),
                    ]
                    y = resultado.shape[0] // 2 - 110
                    for texto, cor, escala, espessura in linhas_texto:
                        (tw, th), _ = cv2.getTextSize(
                            texto, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
                        x = (resultado.shape[1] - tw) // 2
                        cv2.putText(resultado, texto, (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura)
                        y += th + 22

                    cv2.imshow("Calibracao Fujinon 35mm [Hikrobot]", resultado)
                    cv2.waitKey(5000)
                    break

                else:
                    print("[ERRO] Calibracao falhou. Tira mais fotos em posicoes diferentes.")

            elif tecla == ord('q'):
                print("\n[SAIR] Calibracao cancelada.")
                break

    except KeyboardInterrupt:
        print("\n[SAIR] Ctrl+C detectado - a fechar...")

    finally:
        cam.fechar()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────
#  PONTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    calibrar()

import cv2
import numpy as np
import os
import glob
import sys
import time
import socket
import ctypes
from ctypes import *
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
    print("[ERRO] Nao foi possivel importar o SDK da Hikrobot.")
    sys.exit(1)

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
CAMERA_IP       = "192.168.2.30"

# --- Parâmetros da câmara (fixar evita oscilação entre frames) ---
# Ajusta EXPOSURE_US até a imagem ficar nítida e estável. Coloca a None
# qualquer um destes para deixar a câmara decidir (auto).
CAMERA_EXPOSURE_US = 30000.0   # microsegundos. DEVE ser igual ao capturador (PSA_Fotos)
CAMERA_GAIN        = 10.0      # dB. DEVE ser igual ao capturador
CAMERA_FRAME_RATE  = 15.0      # fps alvo (limita carga e estabiliza)

XADREZ_COLS  = 7
XADREZ_ROWS  = 7
TAMANHO_QUAD = 25  # mm
FICHEIRO_CALIBRACAO = "calibracao_camera_teste.npz"

PASTA_MODELOS = r"C:\Users\j0a0p\Desktop\PSA_G4_AV\Fotos_Asa_teste"

# Interruptor da retificacao. False = imagem original (filtros p/ essa resolucao).
USAR_RETIFICACAO = False

AREA_MINIMA_OBJETO      = 2000   # baixado de 5000: asas 2 e 3 sao finas (area pequena)
INTERVALO_PROCESSAMENTO = 0.05
DISTANCIA_MINIMA_NOVA   = 300
TEMPO_EXPIRACAO         = 1.0    # 4s: equilibrio entre nao piscar e apagar
                                 # depressa quando a peca sai do tapete

# --- Segmentação (fundo ESCURO, asas CLARAS) ---
# Pixéis mais claros que THRESH_VALOR são considerados objeto.
# Se 'None', usa Otsu adaptativo como reserva. Começa fixo: muito mais estável.
THRESH_VALOR        = 70     # 0–255. Sobe se apanha o fundo, desce se perde a asa
SOLIDITY_MINIMA     = 0.12   # baixado de 0.30: ganchos finos/abertos tem solidity
                             # muito baixa (quase uma linha curva). Usa a tecla 'd'
                             # para ver a solidity real e afinar.
SCORE_MAXIMO_MATCH  = 4.0    # nova escala (comprimento dominante): asa CERTA da
                             # score < 3, asas ERRADAS dao > 5. 4.0 separa bem.
                             # Sobe se asa certa falhar poses; desce se aceitar
                             # asa errada. Usa a tecla 'd' (mostra score por tipo).

# --- Motor de identificação melhorado (Opção 1: forma + descritores) ---
# Peso dos descritores de forma (aspect/circularity/solidity) no score final.
# Ajuda a rejeitar as OUTRAS asas-gancho parecidas. 0 = só matchShapes.
PESO_DESCRITORES    = 1.5
# A face (Normal/Invertida) é decidida comparando o alvo contra os modelos
# de CADA face. Se os scores das duas faces ficarem mais próximos que esta
# margem, a pose é ambígua e fica-se pela melhor.
MARGEM_QUIRALIDADE  = 0.10
# Template matching (OFF nesta versao; precisa de modelos com textura)
USAR_TEMPLATE       = False

CORES_PECAS = {
    "Asa_0": (0, 255, 0),
    "Asa_1": (0, 200, 255),
    "Asa_2": (255, 100, 0),
    "Asa_3": (0, 100, 255),
}

# =============================================================================
# OFFSETS DO GRIPPER (em mm e graus)
# Estes valores são somados diretamente ao X, Y e Rz do centro de massa 
# ANTES de serem enviados para o robô.
# =============================================================================
OFFSETS_GRIPPER = {
    "Asa_0_Normal":    {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": 0.0},
    "Asa_0_Invertida": {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": 180.0},
    
    "Asa_1_Normal":    {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": -2.5},
    "Asa_1_Invertida": {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": 187.5},
    
    "Asa_2_Normal":    {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": -2.5},
    "Asa_2_Invertida": {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": 187.5},
    
    "Asa_3_Normal":    {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": -2.5},
    "Asa_3_Invertida": {"offset_x": 0.0, "offset_y": 0.0, "offset_rz": 187.5},
}

ASA_SELECIONADA = None

# --- Robot ---
ROBOT_IP   = "192.168.0.16"
ROBOT_PORT = 5890
ROBOT_TIMEOUT = 120
# =============================================================================

# =============================================================================
# MENU DE SELEÇÃO DE ASA
# =============================================================================

def menu_selecao_asa():
    if not os.path.exists(PASTA_MODELOS):
        print(f"[ERRO] Pasta de modelos nao encontrada: {PASTA_MODELOS}")
        sys.exit(1)

    pastas = [d for d in os.listdir(PASTA_MODELOS) if os.path.isdir(os.path.join(PASTA_MODELOS, d))]
    
    asas_base = set()
    for p in pastas:
        if "_Normal" in p: asas_base.add(p.replace("_Normal", ""))
        elif "_Invertida" in p: asas_base.add(p.replace("_Invertida", ""))
        else: asas_base.add(p) 
        
    asas_disponiveis = sorted(list(asas_base))

    if not asas_disponiveis:
        print(f"[ERRO] Nenhuma asa encontrada em: {PASTA_MODELOS}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  SISTEMA DE VISAO — SELECAO DE ASA")
    print("=" * 50 + "\n")
    print("  Asas disponiveis para monitorizar:\n")
    
    for i, nome in enumerate(asas_disponiveis):
        print(f"    [{i}] {nome} (Deteta faces Normais e Invertidas)")

    print("\n  Digite o numero da asa e prima ENTER.")
    print("=" * 50)

    while True:
        try:
            escolha = input("  Opcao: ").strip()
            idx = int(escolha)
            if 0 <= idx < len(asas_disponiveis):
                asa = asas_disponiveis[idx]
                print(f"\n  [OK] Asa selecionada: {asa}")
                print("=" * 50 + "\n")
                return asa
            else:
                print(f"  [!] Opcao invalida. Escolhe entre 0 e {len(asas_disponiveis) - 1}.")
        except ValueError:
            print("  [!] Introduz um numero valido.")

# =============================================================================
# CLASSE: ROBOT TM5-700
# =============================================================================

class RobotTM5:
    def __init__(self):
        self.socket  = None
        self.ligado  = False
        self.msg_id  = 1
        self.ultimo_reconnect = 0.0

    def calcular_checksum(self, dados):
        cs = 0
        for c in dados: cs ^= ord(c)
        return format(cs, '02X')

    def construir_pacote(self, script):
        dados  = f"{self.msg_id},{script}"
        tam    = len(dados.encode('utf-8'))
        corpo  = f"TMSCT,{tam},{dados},"
        cs     = self.calcular_checksum(corpo)
        pacote = f"${corpo}*{cs}\r\n"
        self.msg_id += 1
        return pacote

    def ligar_silencioso(self):
        agora = time.time()
        if agora - self.ultimo_reconnect < 2.0: return
        self.ultimo_reconnect = agora

        try:
            if self.socket:
                try: self.socket.close()
                except: pass
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(0.05)
            self.socket.connect((ROBOT_IP, ROBOT_PORT))
            self.ligado = True
            
            try:
                msg = self.socket.recv(1024).decode('utf-8').strip()
                print(f"  [OK] Robot reconectado — Listen Node ativo: {msg}")
            except socket.timeout: pass
        except Exception:
            self.ligado = False

    def ligar(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((ROBOT_IP, ROBOT_PORT))
            self.ligado = True
            print(f"  [OK] Robot TM5-700 ligado: {ROBOT_IP}:{ROBOT_PORT}")
            try:
                msg = self.socket.recv(1024).decode('utf-8').strip()
                print(f"  [OK] Listen Node ativo: {msg}")
            except Exception: pass
            return True
        except Exception as e:
            print(f"  [AVISO] Nao foi possivel ligar ao robot: {e}")
            self.ligado = False
            return False

    def enviar_dados_asa(self, tipo_asa, cx, cy, rz, n_asas, invertida):
        if not self.ligado: return False

        # var_invertida: 0 = Normal, 1 = Invertida (inteiro -> if simples no TMflow)
        script = (
            f"var_tipo_asa = {tipo_asa}\r\n"
            f"var_x = {cy}\r\n"
            f"var_y = {-cx}\r\n"
            f"var_rz = {rz}\r\n"
            f"var_n_asas = {n_asas}\r\n"
            f"var_invertida = {invertida}\r\n"
            f"ScriptExit()"
        )
        pacote = self.construir_pacote(script)

        try:
            self.socket.sendall(pacote.encode('utf-8'))
            self.socket.settimeout(0.02)
            try: self.socket.recv(1024)
            except socket.timeout: pass
            return True
        except Exception:
            self._reconectar()
            return False

    def _reconectar(self):
        self.ligado = False
        if self.socket:
            try: self.socket.close()
            except: pass

    def desligar(self):
        if self.socket:
            try: self.socket.close()
            except: pass
        self.ligado = False

# =============================================================================
# REGISTO DE PEÇAS
# =============================================================================
class RegistoPecas:
    def __init__(self, robot=None):
        self.pecas    = {}
        self.contador = 0
        self.robot    = robot
        self.total_pecas_historico = 0 

    def limpar_expiradas(self):
        agora     = time.time()
        expiradas = [pid for pid, p in self.pecas.items() if agora - p["timestamp"] > TEMPO_EXPIRACAO]
        for pid in expiradas:
            print(f"  [~] Peca #{pid:04d} ({self.pecas[pid]['nome']}) saiu do tapete.")
            del self.pecas[pid]

    def _peca_mais_proxima(self, centro_px):
        melhor_id, melhor_dist = None, float('inf')
        centro = np.array(centro_px)
        for pid, p in self.pecas.items():
            dist = np.linalg.norm(centro - np.array(p["centro_px"]))
            if dist < melhor_dist:
                melhor_dist = dist
                melhor_id   = pid
        return melhor_id, melhor_dist

    def peca_mais_esquerda(self):
        if not self.pecas: return None
        return min(self.pecas.values(), key=lambda p: p["centro_coord"][0])

    def enviar_estado_robot(self, asa_selecionada):
        if self.robot is None or not self.pecas: return

        try:
            tipo_asa = int(asa_selecionada.split("_")[1])
        except Exception:
            tipo_asa = 0

        peca_esq = self.peca_mais_esquerda()
        cx, cy   = peca_esq["centro_coord"]
        rz       = peca_esq["angulo"]
        n_asas   = len(self.pecas)
        nome_peca = peca_esq["nome"]

        # 0 = Normal, 1 = Invertida (lido a partir do nome decidido na identificacao)
        invertida = 1 if "Invertida" in nome_peca else 0

        # =================================================================
        # LÓGICA DE OFFSETS
        # X/Y: fixos para todas as asas. ROTACAO: depende da FACE (Normal soma
        # +180, Invertida soma 0), lida do dicionario OFFSETS_GRIPPER pelo nome
        # da peca (ex: "Asa_2_Normal"). E isto que faz a Normal e a Invertida
        # chegarem ao robo com angulos diferentes.
        # =================================================================
        offset_rz = OFFSETS_GRIPPER.get(nome_peca, {}).get("offset_rz", 0.0)

        # Aplica o offset ao Centro de Massa (em mm e graus)
        cx += -170
        cy += 115
        rz += offset_rz

        # Garante que a rotação final se mantém sempre entre 0 e 360 graus
        rz = rz % 360
        rz = round(rz, 2)
        # =================================================================

        if not self.robot.ligado: self.robot.ligar_silencioso()
        if self.robot.ligado:
            # Envia as coordenadas FINAIS já com o offset aplicado
            self.robot.enviar_dados_asa(tipo_asa, cx, cy, rz, n_asas, invertida)

    def atualizar(self, centro_px, nome, centro_coord, angulo, score, box, asa_selecionada):
        # NOTA: aceitamos tambem deteccoes "Desconhecida" para REFRESCAR o
        # timestamp de uma peca ja registada que continua no mesmo sitio. Assim
        # ela NAO "sai do tapete" so porque o reconhecimento falhou num frame
        # (o matchShapes oscila a volta do limiar por causa da pose 3D).
        pid_prox, dist_prox = self._peca_mais_proxima(centro_px)

        if pid_prox is not None and dist_prox <= DISTANCIA_MINIMA_NOVA:
            # Mesma peca (mesma posicao). Refresca sempre o timestamp.
            atualizacao = {
                "centro_px":    centro_px,
                "centro_coord": centro_coord,
                "angulo":       angulo,
                "box":          box,
                "timestamp":    time.time(),
            }
            # So atualiza o nome/score se ESTE frame reconheceu algo (!= Desconhecida).
            # Se falhou o reconhecimento, mantem o ultimo nome valido.
            if nome != "Desconhecida":
                atualizacao["nome"]  = nome
                atualizacao["score"] = score
            self.pecas[pid_prox].update(atualizacao)
        else:
            # Posicao nova. So cria peca se for um reconhecimento valido
            # (nao queremos criar pecas a partir de deteccoes desconhecidas/ruido).
            if nome == "Desconhecida":
                return
            self.contador += 1
            self.total_pecas_historico += 1

            self.pecas[self.contador] = {
                "centro_px": centro_px, "nome": nome, "timestamp": time.time(),
                "centro_coord": centro_coord, "angulo": angulo, "score": score, "box": box,
            }
            cx, cy = centro_coord
            print(f"  [+] NOVA #{self.contador:04d} | {nome} | X={cx:.1f} Y={cy:.1f} | Rot={int(angulo)}° | Total Histórico: {self.total_pecas_historico}")

# =============================================================================
# CÂMARA HIKROBOT
# =============================================================================

class CameraHikrobot:
    def __init__(self):
        self.cam        = MvCamera()
        self.stOutFrame = MV_FRAME_OUT()
        memset(byref(self.stOutFrame), 0, sizeof(self.stOutFrame))
        self.buf_convert     = None   # buffer reutilizável p/ conversão
        self.buf_convert_len = 0

    def _set_exposicao(self):
        # Desliga auto-exposição/ganho e fixa valores -> frames estáveis.
        try:
            if CAMERA_EXPOSURE_US is not None:
                self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # Off
                self.cam.MV_CC_SetFloatValue("ExposureTime", float(CAMERA_EXPOSURE_US))
            if CAMERA_GAIN is not None:
                self.cam.MV_CC_SetEnumValue("GainAuto", 0)      # Off
                self.cam.MV_CC_SetFloatValue("Gain", float(CAMERA_GAIN))
            if CAMERA_FRAME_RATE is not None:
                self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(CAMERA_FRAME_RATE))
            print("[OK] Exposicao/ganho fixados (sem auto).")
        except Exception as e:
            print(f"[AVISO] Nao foi possivel fixar exposicao: {e}")

    def ligar(self):
        MvCamera.MV_CC_Initialize()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            print("[ERRO] Nenhuma camara GigE encontrada.")
            return False

        cam_idx = None
        for i in range(deviceList.nDeviceNum):
            info   = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            ip     = info.SpecialInfo.stGigEInfo.nCurrentIp
            ip_str = f"{(ip & 0xff000000) >> 24}.{(ip & 0x00ff0000) >> 16}.{(ip & 0x0000ff00) >> 8}.{ip & 0x000000ff}"
            if ip_str == CAMERA_IP:
                cam_idx = i
                break

        if cam_idx is None: return False

        stDevInfo = cast(deviceList.pDeviceInfo[cam_idx], POINTER(MV_CC_DEVICE_INFO)).contents
        if self.cam.MV_CC_CreateHandle(stDevInfo) != 0: return False
        if self.cam.MV_CC_OpenDevice()             != 0: return False

        nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
        if nPacketSize > 0: self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        self.cam.MV_CC_SetEnumValue("TriggerMode",    MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetEnumValue("AcquisitionMode", 2)
        self._set_exposicao()

        print(f"[OK] Camara {CAMERA_IP} ligada.")
        return True

    def iniciar_captura(self):
        return self.cam.MV_CC_StartGrabbing() == 0

    def capturar_frame_direto(self):
        # Timeout curto: se um frame falha, tenta o próximo em vez de bloquear 1s.
        if self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 200) != 0:
            return None
        try:
            info = self.stOutFrame.stFrameInfo
            w, h = int(info.nWidth), int(info.nHeight)

            # Caminho rápido: já é mono ou RGB direto.
            if info.enPixelType == PixelType_Gvsp_Mono8:
                raw = ctypes.string_at(self.stOutFrame.pBufAddr, w * h)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            # Conversão OFICIAL do SDK -> BGR8. Robusto p/ qualquer Bayer/YUV/packed.
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
# RECONHECIMENTO E CALIBRAÇÃO 
# =============================================================================
base_de_dados_asas = {}

# =============================================================================
# SEGMENTAÇÃO ÚNICA (usada por modelos E tempo real -> consistência total)
# =============================================================================
def _para_cinza_fundo_preto(img):
    """Converte qualquer imagem para cinzento sobre fundo PRETO.
    Se vier com alpha (PNG do rembg), compoe sobre preto usando o alpha como
    mascara -> a peca recortada fica clara sobre preto, igual ao tapete ao vivo.
    Assim o MESMO threshold serve para modelos e para producao."""
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr   = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        composto = (bgr * alpha).astype(np.uint8)   # fora da peca -> preto
        return cv2.cvtColor(composto, cv2.COLOR_BGR2GRAY)
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # ja e cinzento

def segmentar(img, aplicar_morfologia=True):
    """Segmentacao binaria unica. THRESH_VALOR fixo (fundo escuro, peca clara).
    Devolve a mascara binaria. Identica para modelos e tempo real."""
    gray = _para_cinza_fundo_preto(img)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    if THRESH_VALOR is not None:
        _, thresh = cv2.threshold(blur, THRESH_VALOR, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if aplicar_morfologia:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh

# =============================================================================
# DESCRITORES DE FORMA
# =============================================================================
def descritores_forma(contorno):
    """Descritores invariantes a escala/translacao E a rotacao. Usados para
    distinguir a asa escolhida das outras asas-gancho parecidas.

    Os descritores 'comprimento' e 'espessura' foram os que provaram (na
    analise das fotos) separar melhor a Asa_2 da Asa_3:
      - comprimento (perim/2): a Asa_3 e mais alongada -> d'=18 vs Asa_2.
      - espessura (area/(perim/2)): quao 'grosso' e o fio da peca.
    Sao ~invariantes a rotacao e a face (Normal/Invertida tem o mesmo valor),
    por isso servem para o TIPO de asa, nao para a face."""
    area = abs(cv2.contourArea(contorno))
    if area <= 0:
        return None
    peri = cv2.arcLength(contorno, True)
    hull = cv2.convexHull(contorno)
    area_hull = cv2.contourArea(hull)
    rect = cv2.minAreaRect(contorno)
    (rw, rh) = rect[1]
    aspect = (max(rw, rh) / min(rw, rh)) if min(rw, rh) > 1e-3 else 0.0
    circ   = (4 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
    sol    = (area / area_hull) if area_hull > 0 else 0.0
    comprimento = peri / 2.0
    espessura   = area / (peri / 2.0) if peri > 0 else 0.0
    return {"aspect": aspect, "circular": circ, "solidity": sol, "area": area,
            "comprimento": comprimento, "espessura": espessura}

def carregar_modelos_de_imagem(asa_base):
    sucesso = False
    variacoes = [f"{asa_base}_Normal", f"{asa_base}_Invertida", asa_base]

    for var in variacoes:
        caminho_subpasta = os.path.join(PASTA_MODELOS, var)
        if not os.path.isdir(caminho_subpasta): continue

        modelos_tipo = []   # agora lista de dicts: {contorno, desc}
        ficheiros = glob.glob(os.path.join(caminho_subpasta, "*.png")) + glob.glob(os.path.join(caminho_subpasta, "*.jpg"))

        for arquivo_img in ficheiros:
            img = cv2.imread(arquivo_img, cv2.IMREAD_UNCHANGED)
            if img is None: continue

            # MESMA segmentacao do tempo real. Ignora o alpha como mascara de
            # corte: compoe sobre preto e aplica o threshold fixo. Assim uma
            # foto recortada (rembg) e uma foto crua de fundo preto dao o MESMO
            # tipo de contorno -> consistencia total com a producao.
            thresh = segmentar(img, aplicar_morfologia=True)

            contornos = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            if contornos:
                maior = max(contornos, key=cv2.contourArea)
                if cv2.contourArea(maior) > 500:
                    desc = descritores_forma(maior)
                    if desc:
                        modelos_tipo.append({"contorno": maior, "desc": desc})

        if modelos_tipo:
            base_de_dados_asas[var] = modelos_tipo
            print(f"  [OK] Modelos: {var} ({len(modelos_tipo)} fotos)")
            sucesso = True

    return sucesso

def extrair_contornos(frame):
    # MESMA segmentacao dos modelos (funcao unica) -> contornos comparaveis.
    thresh = segmentar(frame, aplicar_morfologia=True)

    contornos = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    contornos_validos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < AREA_MINIMA_OBJETO:
            continue
        # Solidity = area / area do convex hull. Rejeita ruido/reflexos
        # esburacados sem afetar pecas solidas.
        hull = cv2.convexHull(c)
        area_hull = cv2.contourArea(hull)
        solidity = area / area_hull if area_hull > 0 else 0
        if solidity < SOLIDITY_MINIMA:
            continue
        contornos_validos.append(c)

    contornos_validos.sort(key=cv2.contourArea, reverse=True)
    return contornos_validos, thresh

def _score_forma(contorno_alvo, desc_alvo, modelo):
    """Score de SEMELHANCA DE FORMA (ignora face). Menor = mais parecido.

    REEQUILIBRADO apos analise dos dados reais: classificar pelo COMPRIMENTO
    sozinho dava 4/4 acertos nas asas-gancho, enquanto o matchShapes misturado
    dava 2/4 (instavel com a pose, abafava o sinal limpo do comprimento).
    Por isso: o comprimento DOMINA, os outros descritores apoiam, e o
    matchShapes entra com peso minimo (so para casos onde a forma global
    ajuda a desempatar). A Asa_1/2/3 distinguem-se sobretudo pelo comprimento.
    """
    cnt_modelo = modelo["contorno"]
    desc_mod   = modelo["desc"]

    d_forma = cv2.matchShapes(contorno_alvo, cnt_modelo, cv2.CONTOURS_MATCH_I1, 0.0)

    ESCALA_COMPRIMENTO = 300.0    # mais sensivel (era 1000): diferencas de
                                  # ~250 entre A2(1740) e A3(2010) viram ~0.8
    ESCALA_ESPESSURA   = 60.0

    def dif_norm(chave, escala):
        a = desc_alvo.get(chave); m = desc_mod.get(chave)
        if a is None or m is None:
            return 0.0
        return abs(a - m) / escala

    d_desc = (
        dif_norm("comprimento", ESCALA_COMPRIMENTO) * 6.0 +   # DOMINA
        dif_norm("espessura",   ESCALA_ESPESSURA)   * 2.0 +
        abs(desc_alvo["circular"] - desc_mod["circular"]) * 3.0 +
        abs(desc_alvo["solidity"] - desc_mod["solidity"]) * 2.0 +
        abs(desc_alvo["aspect"]   - desc_mod["aspect"])   * 0.10
    )

    # matchShapes com peso MINIMO (era o fator dominante, agora so apoia)
    return d_forma * 0.3 + d_desc

def face_pela_zona_reta(contorno):
    """Para pecas tipo ANEL (Asa_0), que nao tem ponta afilada: usa a zona
    mais RETA do contorno (a parte da cola) como referencia de assimetria.
    Ancora a orientacao pela assimetria (skewness) da forma, o que torna a
    deteccao robusta a rotacao. Normal e Invertida -> lados opostos."""
    pts = contorno.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 20:
        return 0
    centro = pts.mean(axis=0)
    pts_c = pts - centro
    cov = np.cov(pts_c.T)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0
    eixo1 = evecs[:, np.argmax(evals)]
    eixo2 = evecs[:, np.argmin(evals)]
    proj1 = pts_c @ eixo1
    proj2 = pts_c @ eixo2
    # ancora o sentido do eixo principal pela 3a potencia (assimetria)
    if np.mean(proj1**3) < 0:
        proj1 = -proj1
    # curvatura local (0 = reto)
    k = max(3, n // 30)
    curv = np.zeros(n)
    for i in range(n):
        a = pts[(i - k) % n]; b = pts[i]; c = pts[(i + k) % n]
        v1 = b - a; v2 = c - b
        nn1 = np.linalg.norm(v1); nn2 = np.linalg.norm(v2)
        if nn1 < 1e-6 or nn2 < 1e-6:
            continue
        curv[i] = np.arccos(np.clip(np.dot(v1, v2) / (nn1 * nn2), -1, 1))
    w = max(5, n // 12)
    soma = np.array([curv[i:i + w].sum() if i + w <= n else
                     curv[i:].sum() + curv[:(i + w) % n].sum() for i in range(n)])
    idx = (int(np.argmin(soma)) + w // 2) % n
    return 1 if proj2[idx] >= 0 else -1

def _sinal_face(contorno, asa_tipo):
    """Escolhe o metodo de assimetria certo conforme o tipo de asa:
    - Asa_0 (anel, sem ponta) -> zona reta da cola.
    - Outras (ganchos) -> ponta afilada."""
    if asa_tipo == "Asa_0":
        return face_pela_zona_reta(contorno)
    return face_pela_ponta(contorno)

def face_pela_ponta(contorno):
    """Determina a 'lateralidade' da ponta (extremo afilado) apos alinhar a
    peca pelo eixo principal (PCA). Devolve +1 ou -1.
    Invariante a rotacao: a face Normal e a Invertida (espelhada) dao sinais
    OPOSTOS, esteja a peca rodada como estiver. E a forma mais fiavel de
    distinguir face nestas asas, pois usa a assimetria real (a ponta)."""
    pts = contorno.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return 0
    centro = pts.mean(axis=0)
    pts_c = pts - centro
    cov = np.cov(pts_c.T)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0
    eixo1 = evecs[:, np.argmax(evals)]   # comprimento
    eixo2 = evecs[:, np.argmin(evals)]   # lateral
    proj1 = pts_c @ eixo1
    proj2 = pts_c @ eixo2
    dists = np.linalg.norm(pts_c, axis=1)
    idx_ponta = int(np.argmax(dists))    # a ponta = ponto mais distante do centro
    lado = proj2[idx_ponta]
    # remove ambiguidade do sinal do eixo PCA: ancora pela cauda comprida
    if proj1[idx_ponta] < 0:
        lado = -lado
    return 1 if lado >= 0 else -1

def angulo_pela_ponta(contorno):
    """Angulo da peca SEM ambiguidade de 180 graus.
    O minAreaRect da um angulo da CAIXA, que e simetrica -> nao distingue a
    peca da peca virada ao contrario (ambiguidade de 180). Aqui usamos a
    direcao do centroide ate a PONTA (ponto mais distante), que e um ponto
    unico da peca, logo o angulo e inequivoco. Resolve o '180 a mais'."""
    pts = contorno.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return None
    centro = pts.mean(axis=0)
    pts_c = pts - centro
    dists = np.linalg.norm(pts_c, axis=1)
    idx_ponta = int(np.argmax(dists))
    vetor = pts_c[idx_ponta]
    ang = np.degrees(np.arctan2(vetor[1], vetor[0]))
    return ang % 360

def _sinal_face_modelos(var, asa_tipo):
    """Sinal medio da assimetria nos modelos de uma face, usando o metodo
    adequado ao tipo (zona reta p/ anel, ponta p/ ganchos)."""
    modelos = base_de_dados_asas.get(var, [])
    if not modelos:
        return None
    sinais = [_sinal_face(m["contorno"], asa_tipo) for m in modelos]
    sinais = [s for s in sinais if s != 0]
    if not sinais:
        return None
    return np.sign(np.mean(sinais))

def decidir_face(contorno_alvo, asa_tipo):
    """Decide Normal vs Invertida pela assimetria (ponta p/ ganchos, zona reta
    p/ anel), comparando o sinal do alvo com o sinal tipico de cada face."""
    nome_n = f"{asa_tipo}_Normal"
    nome_i = f"{asa_tipo}_Invertida"
    sinal_alvo = _sinal_face(contorno_alvo, asa_tipo)
    sinal_n = _sinal_face_modelos(nome_n, asa_tipo)
    sinal_i = _sinal_face_modelos(nome_i, asa_tipo)

    if sinal_n is None and sinal_i is None:
        return asa_tipo
    if sinal_n is None:
        return nome_i if sinal_alvo == sinal_i else nome_n
    if sinal_i is None:
        return nome_n if sinal_alvo == sinal_n else nome_i
    if sinal_n == sinal_i:
        return nome_n
    return nome_n if sinal_alvo == sinal_n else nome_i

def _melhor_score_face(contorno_alvo, desc_alvo, var):
    """Menor score de forma do alvo contra TODOS os modelos de uma dada face.
    Devolve inf se a face nao existir."""
    modelos = base_de_dados_asas.get(var, [])
    melhor = float('inf')
    for modelo in modelos:
        s = _score_forma(contorno_alvo, desc_alvo, modelo)
        if s < melhor:
            melhor = s
    return melhor

def _asas_disponiveis():
    """Lista os tipos de asa (Asa_0, Asa_1, ...) que tem modelos carregados,
    a partir das chaves de base_de_dados_asas (que sao Asa_X_Normal etc.)."""
    tipos = set()
    for chave in base_de_dados_asas.keys():
        partes = chave.split("_")
        if len(partes) >= 2:
            tipos.add("_".join(partes[:2]))   # "Asa_2_Normal" -> "Asa_2"
        else:
            tipos.add(chave)
    return sorted(tipos)

def _score_e_face_de_tipo(contorno_alvo, desc_alvo, asa_tipo):
    """Para um TIPO de asa, devolve (melhor_score, melhor_nome_com_face)
    comparando contra as faces Normal/Invertida/base desse tipo."""
    nomes = [f"{asa_tipo}_Normal", f"{asa_tipo}_Invertida", asa_tipo]
    melhor_s, melhor_nome = float('inf'), None
    for nome in nomes:
        s = _melhor_score_face(contorno_alvo, desc_alvo, nome)
        if s < melhor_s:
            melhor_s, melhor_nome = s, nome
    return melhor_s, melhor_nome

def identificar_asa(contorno_alvo, asa_base):
    """NOVA LOGICA (compara contra as 4 asas):
    1. Calcula o score do alvo contra CADA tipo de asa carregado.
    2. A asa de MENOR score e a 'vencedora' (a mais parecida).
    3. So devolve um nome valido se a vencedora for a asa ESCOLHIDA (asa_base)
       E o score estiver abaixo de SCORE_MAXIMO_MATCH.
       Caso contrario devolve 'Desconhecida' -> nao marca, nao envia ao robo.

    Assim, uma Asa_3 no tapete (com Asa_2 escolhida) NAO e aceite como Asa_2,
    porque a vencedora e a Asa_3, que nao e a escolhida."""
    desc_alvo = descritores_forma(contorno_alvo)
    if desc_alvo is None:
        return "Desconhecida", float('inf')

    tipos = _asas_disponiveis()
    if not tipos:
        return "Desconhecida", float('inf')

    # Score contra cada tipo
    resultados = []  # (score, tipo, nome_com_face)
    for tipo in tipos:
        s, nome = _score_e_face_de_tipo(contorno_alvo, desc_alvo, tipo)
        if s < float('inf'):
            resultados.append((s, tipo, nome))

    if not resultados:
        return "Desconhecida", float('inf')

    resultados.sort(key=lambda x: x[0])
    melhor_score, melhor_tipo, melhor_nome = resultados[0]

    # Rejeita se nem sequer parece uma asa (score alto demais)
    if melhor_score >= SCORE_MAXIMO_MATCH:
        return "Desconhecida", melhor_score

    # So aceita se a asa mais parecida for a ESCOLHIDA
    if melhor_tipo != asa_base:
        # E mais parecida com outra asa -> nao e a que queremos
        return "Desconhecida", melhor_score

    # E a asa escolhida. A FACE (Normal/Invertida) e decidida pela ponta
    # alinhada por PCA (o score por forma NAO distingue face, pois os
    # descritores sao invariantes ao espelhamento).
    nome_com_face = decidir_face(contorno_alvo, melhor_tipo)
    return nome_com_face, melhor_score

def pixels_para_mm(ponto_px, escala_mm_por_px):
    x_px, y_px = ponto_px
    if escala_mm_por_px is None: return (x_px, y_px)
    return (round(x_px * escala_mm_por_px, 2), round(y_px * escala_mm_por_px, 2))

def aplicar_calibracao(frame, camera_matrix, dist_coeffs):
    if camera_matrix is None or dist_coeffs is None: return frame
    h, w = frame.shape[:2]
    nova_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    frame_corrigido = cv2.undistort(frame, camera_matrix, dist_coeffs, None, nova_matrix)
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        frame_corrigido = frame_corrigido[y:y + h_roi, x:x + w_roi]
        frame_corrigido = cv2.resize(frame_corrigido, (w, h))
    return frame_corrigido

def carregar_calibracao():
    if os.path.exists(FICHEIRO_CALIBRACAO):
        dados = np.load(FICHEIRO_CALIBRACAO)
        return dados["camera_matrix"], dados["dist_coeffs"]
    return None, None

def calibrar_escala(hikrobot, camera_matrix, dist_coeffs):
    print("\n" + "=" * 50)
    print("  CALIBRACAO DE ESCALA (Pixeis -> mm)")
    print("  Pousa o tabuleiro PLANO no tapete.")
    print("  ESPACO = confirmar  |  Q = saltar")
    print("=" * 50)
    escala_final = None

    while True:
        frame = hikrobot.capturar_frame_direto()
        if frame is None:
            time.sleep(0.1)
            continue

        frame = aplicar_calibracao(frame, camera_matrix, dist_coeffs)
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        encontrado, cantos_ref = cv2.findChessboardCornersSB(gray, (XADREZ_COLS, XADREZ_ROWS), flags)

        escala_preview = None
        if encontrado:
            cv2.drawChessboardCorners(frame_display, (XADREZ_COLS, XADREZ_ROWS), cantos_ref, encontrado)
            distancias_px = []
            c_res = cantos_ref.reshape(XADREZ_ROWS, XADREZ_COLS, 2)
            for row in range(XADREZ_ROWS):
                for col in range(XADREZ_COLS - 1):
                    distancias_px.append(np.linalg.norm(c_res[row, col] - c_res[row, col + 1]))
            for row in range(XADREZ_ROWS - 1):
                for col in range(XADREZ_COLS):
                    distancias_px.append(np.linalg.norm(c_res[row, col] - c_res[row + 1, col]))
            
            escala_preview = TAMANHO_QUAD / np.mean(distancias_px)
            cv2.putText(frame_display, f"Escala: {escala_preview:.4f} mm/px | ESPACO para guardar", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame_display, "A procurar cantos...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

        h_real, w_real = frame_display.shape[:2]
        proporcao = w_real / h_real
        display_width = 1024
        display_height = int(display_width / proporcao)

        cv2.imshow("Calibracao Escala", cv2.resize(frame_display, (display_width, display_height)))
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):
            if escala_preview is not None:
                escala_final = escala_preview
                print(f"[OK] Escala definida: {escala_final:.4f} mm/px")
                break
        elif tecla == ord('q'): break

    cv2.destroyWindow("Calibracao Escala")
    return escala_final

# =============================================================================
# MAIN
# =============================================================================
def carregar_todas_as_asas():
    """Carrega os modelos de TODAS as asas que existam em PASTA_MODELOS.
    Necessario para a identificacao comparar o alvo contra todas e escolher
    a mais parecida (em vez de aceitar qualquer gancho como a asa escolhida)."""
    if not os.path.isdir(PASTA_MODELOS):
        print(f"[ERRO] Pasta de modelos nao existe: {PASTA_MODELOS}")
        return False
    # Descobre os tipos de asa pelas subpastas (Asa_0, Asa_1, ...)
    tipos = set()
    for d in os.listdir(PASTA_MODELOS):
        if os.path.isdir(os.path.join(PASTA_MODELOS, d)) and d.startswith("Asa_"):
            partes = d.split("_")
            tipos.add("_".join(partes[:2]) if len(partes) >= 2 else d)
    if not tipos:
        print(f"[ERRO] Nenhuma subpasta Asa_X em {PASTA_MODELOS}")
        return False
    algum = False
    for tipo in sorted(tipos):
        if carregar_modelos_de_imagem(tipo):
            algum = True
    if algum:
        print(f"  [OK] Asas carregadas para comparacao: {', '.join(_asas_disponiveis())}")
    return algum

if __name__ == "__main__":
    # Carrega TODAS as asas (para comparar e nao confundir), mas o ALVO que o
    # robo vai buscar e o escolhido no menu.
    if not carregar_todas_as_asas(): sys.exit(1)
    ASA_SELECIONADA = menu_selecao_asa()

    robot = RobotTM5()
    robot.ligar()

    camara = CameraHikrobot()
    if not camara.ligar() or not camara.iniciar_captura():
        sys.exit(1)

    registo_tapete = RegistoPecas(robot)
    ultimo_processamento = 0.0

    try:
        mtx, dist    = carregar_calibracao()
        escala_mm_px = calibrar_escala(camara, mtx, dist)
        unidade      = "mm" if escala_mm_px is not None else "px"
        cor_asa      = CORES_PECAS.get(ASA_SELECIONADA, (0, 255, 0))

        # Retificador (lente + perspetiva). Substitui o aplicar_calibracao antigo
        # (que so corrigia a lente). Mesma retificacao que o capturador de fotos,
        # garantindo que modelos e tempo real sao processados de forma identica.
        # Retificador (lente + perspetiva). So ativo se USAR_RETIFICACAO=True.
        retificador = Retificador() if USAR_RETIFICACAO else None
        if retificador is None:
            print("[INFO] Retificacao DESLIGADA -> imagem original.")

        while True:
            frame = camara.capturar_frame_direto()
            if frame is None:
                time.sleep(0.01)
                continue

            agora = time.time()
            frame_final = retificador.aplicar(frame) if retificador is not None else frame
            # Garante um array GRAVAVEL (o buffer da camara pode vir readonly,
            # o que faz o cv2.rectangle/putText falhar com "readonly array").
            if not frame_final.flags.writeable:
                frame_final = frame_final.copy()

            if agora - ultimo_processamento >= INTERVALO_PROCESSAMENTO:
                ultimo_processamento = agora
                registo_tapete.limpar_expiradas()

                contornos_detetados, mascara = extrair_contornos(frame_final)

                for cnt in contornos_detetados:
                    nome, score = identificar_asa(cnt, ASA_SELECIONADA)
                    rect        = cv2.minAreaRect(cnt)
                    box         = np.intp(cv2.boxPoints(rect))
                    centro_px   = (int(rect[0][0]), int(rect[0][1]))
                    centro_mm   = pixels_para_mm(centro_px, escala_mm_px)

                    # ANGULO: para asas-gancho usamos a direcao da ponta
                    # (sem ambiguidade de 180). Para o anel (Asa_0, sem ponta)
                    # mantemos o angulo da caixa, pois nao ha ponta a apontar.
                    ang_ponta = None
                    if not str(nome).startswith("Asa_0"):
                        ang_ponta = angulo_pela_ponta(cnt)

                    if ang_ponta is not None:
                        angulo = ang_ponta
                    else:
                        angulo_raw = rect[2]
                        w_rect, h_rect = rect[1]
                        angulo = (angulo_raw + 90 if w_rect < h_rect else (angulo_raw + 180 if angulo_raw < 0 else angulo_raw))

                    registo_tapete.atualizar(centro_px, nome, centro_mm, angulo, score, box, ASA_SELECIONADA)

                registo_tapete.enviar_estado_robot(ASA_SELECIONADA)

            h_real, w_real = frame_final.shape[:2]
            proporcao = w_real / h_real
            display_width = 1024
            display_height = int(display_width / proporcao)

            # --- DESENHO ---
            for pid, peca in registo_tapete.pecas.items():
                centro, pos_mm, angulo_p = peca["centro_px"], peca["centro_coord"], peca["angulo"]
                
                cor_estado = (0, 165, 255) if "Invertida" in peca["nome"] else cor_asa
                
                cv2.drawContours(frame_final, [peca["box"]], 0, cor_estado, 4)
                cv2.circle(frame_final, centro, 10, cor_estado, -1)
                rad = np.deg2rad(angulo_p)
                pt2 = (int(centro[0] + 100 * np.cos(rad)), int(centro[1] + 100 * np.sin(rad)))
                cv2.line(frame_final, centro, pt2, (0, 0, 255), 4)

                cv2.putText(frame_final, f"#{pid:04d} {peca['nome']}", (centro[0]-80, centro[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_estado, 3)
                cv2.putText(frame_final, f"X:{pos_mm[0]:.1f} Y:{pos_mm[1]:.1f} {unidade}", (centro[0]-80, centro[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame_final, f"Rot: {int(angulo_p)} graus", (centro[0]-80, centro[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.rectangle(frame_final, (0, 0), (w_real, 70), (0, 0, 0), -1)
            cv2.putText(frame_final, f"A reconhecer: {ASA_SELECIONADA} | Pecas no tapete: {len(registo_tapete.pecas)} | T=trocar  M=mascara  D=diagnostico  Q=sair", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor_asa, 3)
            cv2.imshow("Monitorizacao do Tapete", cv2.resize(frame_final, (display_width, display_height)))

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q'): break
            elif tecla == ord('m'):
                # Mostra a mascara binaria -> ajuda a afinar THRESH_VALOR
                _, mascara_dbg = extrair_contornos(frame_final)
                cv2.imshow("Mascara (debug)", cv2.resize(mascara_dbg, (display_width, display_height)))
            elif tecla == ord('d'):
                # DIAGNOSTICO: area e solidity de TODOS os contornos brutos e
                # porque foram aceites/rejeitados. Para afinar AREA_MINIMA e
                # SOLIDITY_MINIMA com as asas finas (2 e 3).
                if len(frame_final.shape) == 3:
                    _g = cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY)
                else:
                    _g = frame_final
                _b = cv2.GaussianBlur(_g, (7, 7), 0)
                _, _th = cv2.threshold(_b, THRESH_VALOR, 255, cv2.THRESH_BINARY)
                _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                _th = cv2.morphologyEx(_th, cv2.MORPH_CLOSE, _k)
                _th = cv2.morphologyEx(_th, cv2.MORPH_OPEN, _k)
                _cnts = cv2.findContours(_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                print(f"\n--- DIAGNOSTICO: {len(_cnts)} contorno(s) bruto(s) ---")
                print(f"    filtros: AREA_MINIMA={AREA_MINIMA_OBJETO}, SOLIDITY_MINIMA={SOLIDITY_MINIMA}, SCORE_MAX={SCORE_MAXIMO_MATCH}")
                for _i, _c in enumerate(sorted(_cnts, key=cv2.contourArea, reverse=True)[:8]):
                    _a = cv2.contourArea(_c)
                    _hl = cv2.contourArea(cv2.convexHull(_c))
                    _s = _a / _hl if _hl > 0 else 0
                    _motivo = "ACEITE"
                    if _a < AREA_MINIMA_OBJETO:
                        _motivo = f"REJEITADO (area {int(_a)} < {AREA_MINIMA_OBJETO})"
                    elif _s < SOLIDITY_MINIMA:
                        _motivo = f"REJEITADO (solidity {_s:.3f} < {SOLIDITY_MINIMA})"
                    # Se passa os filtros, mostra o score contra TODAS as asas
                    _info_score = ""
                    if _motivo == "ACEITE":
                        _nome, _score = identificar_asa(_c, ASA_SELECIONADA)
                        _dsc = descritores_forma(_c)
                        _comp = _dsc["comprimento"] if _dsc else 0
                        _esp = _dsc["espessura"] if _dsc else 0
                        # score contra cada tipo de asa, para ver a margem entre elas
                        _da = descritores_forma(_c)
                        _por_tipo = []
                        for _t in _asas_disponiveis():
                            _st, _ = _score_e_face_de_tipo(_c, _da, _t)
                            _por_tipo.append(f"{_t.replace('Asa_','A')}={_st:.2f}")
                        _info_score = (f" | id={_nome} (compr={_comp:.0f} esp={_esp:.1f})\n"
                                       f"                  scores: {'  '.join(_por_tipo)}")
                    print(f"    contorno {_i}: area={int(_a):8}  solidity={_s:.3f}  -> {_motivo}{_info_score}")
                print("    (a asa vencedora e a de MENOR score. So e aceite se for o alvo E < SCORE_MAX)\n")
            elif tecla == ord('t'):
                cv2.destroyWindow("Monitorizacao do Tapete")
                nova_asa = menu_selecao_asa()
                # Modelos de todas as asas ja estao carregados; trocar so muda
                # o ALVO (qual a asa que o robo vai buscar). Nao recarrega nada.
                if nova_asa != ASA_SELECIONADA:
                    ASA_SELECIONADA = nova_asa
                    cor_asa = CORES_PECAS.get(ASA_SELECIONADA, (0, 255, 0))
                    registo_tapete = RegistoPecas(robot)

    finally:
        robot.desligar()
        camara.parar()
        cv2.destroyAllWindows()
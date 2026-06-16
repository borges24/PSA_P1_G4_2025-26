# -*- coding: utf-8 -*-
"""
============================================================================
 CALIBRACAO DE PERSPETIVA (HOMOGRAFIA) — Projeto PSA  [v2 com deteccao ao vivo]
============================================================================
Faz o plano do tapete parecer visto de cima (top-down), para que uma asa
tenha a MESMA forma no centro e nos cantos do tapete.

NOVIDADES desta versao:
  - DETECCAO AO VIVO: mostra "DETETADO" a verde antes de capturares (como o
    calibrador da lente). Acabou o capturar as cegas.
  - AUTO-GRELHA: tenta varios tamanhos de tabuleiro automaticamente
    (7x7, 7x6, 6x7, 6x6, 6x5, 5x6, 5x5...). Nao precisas de adivinhar
    quantos cantos cabem - ele encontra sozinho.

COMO USAR:
  1. Pousa o tabuleiro PLANO no tapete, na zona de trabalho.
  2. cd para a pasta do projeto e corre: py calibrar_perspetiva.py
  3. Quando aparecer "DETETADO (AxB)" a verde, carrega ESPACO.
  4. Confirma o preview_perspetiva.png (quadrados certos, visto de cima).

REQUER: calibracao_camera_fujinon35.npz (calibracao da lente)
============================================================================
"""
import cv2
import numpy as np
import os
import sys
import ctypes
from ctypes import *

SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
if os.path.exists(SDK_PATH):
    sys.path.insert(0, SDK_PATH)
try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    _SDK = True
except ImportError:
    _SDK = False
    print("[AVISO] SDK nao encontrado. Usa --imagem para uma foto ja tirada.")

# ── CONFIG ──
TAMANHO_QUADRADO = 25     # mm reais de UM quadrado (NAO muda com o nr de cantos)
CALIB_LENTE      = "calibracao_camera_fujinon35.npz"
FICHEIRO_SAIDA   = "perspetiva_tapete.npz"
CAMERA_IP        = "192.168.2.30"
PX_POR_MM        = 4.0

# Exposicao/ganho: MESMOS valores do calibrador da lente (que dava imagem
# clara e bem contrastada). Sem isto a imagem fica escura e o tabuleiro
# nao e detetado por falta de contraste.
EXPOSURE_TIME_US = 40000.0   # 40 ms
GAIN_DB          = 15.0

# Tamanhos de grelha a tentar, por ordem de preferencia (mais cantos primeiro).
# (colunas, linhas) de CANTOS INTERNOS.
GRELHAS_A_TENTAR = [(7, 7), (7, 6), (6, 7), (6, 6), (6, 5), (5, 6), (5, 5), (5, 4), (4, 5), (4, 4)]


def carregar_calib_lente():
    if not os.path.exists(CALIB_LENTE):
        print(f"[ERRO] Falta {CALIB_LENTE}. Corre primeiro a calibracao da lente.")
        sys.exit(1)
    d = np.load(CALIB_LENTE, allow_pickle=True)
    return d["camera_matrix"], d["dist_coeffs"]


def detetar_qualquer_grelha(gray):
    """Tenta detetar varias dimensoes de tabuleiro. Devolve (cols, linhas,
    corners) da primeira que encontrar, ou (None, None, None)."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    for (c, l) in GRELHAS_A_TENTAR:
        found, corners = cv2.findChessboardCorners(gray, (c, l), flags)
        if found:
            return c, l, corners
    return None, None, None


class CamLive:
    def __init__(self):
        self.cam = None
        self.fo = None
        self.buf = None
        self.buflen = 0

    def abrir(self):
        if not _SDK:
            return False
        MvCamera.MV_CC_Initialize()
        lst = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, lst)
        idx = None
        for i in range(lst.nDeviceNum):
            info = cast(lst.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            ip = info.SpecialInfo.stGigEInfo.nCurrentIp
            if f"{(ip>>24)&0xff}.{(ip>>16)&0xff}.{(ip>>8)&0xff}.{ip&0xff}" == CAMERA_IP:
                idx = i; break
        if idx is None:
            print(f"[ERRO] Camara {CAMERA_IP} nao encontrada."); return False
        self.cam = MvCamera()
        info = cast(lst.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam.MV_CC_CreateHandle(info); self.cam.MV_CC_OpenDevice()
        n = self.cam.MV_CC_GetOptimalPacketSize()
        if n > 0: self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", n)
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)
        # Fixar exposicao/ganho -> imagem clara e com contraste (essencial p/ detetar)
        try:
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            self.cam.MV_CC_SetFloatValue("ExposureTime", float(EXPOSURE_TIME_US))
            self.cam.MV_CC_SetEnumValue("GainAuto", 0)
            self.cam.MV_CC_SetFloatValue("Gain", float(GAIN_DB))
        except Exception as e:
            print(f"[AVISO] Nao foi possivel fixar exposicao: {e}")
        self.cam.MV_CC_StartGrabbing()
        self.fo = MV_FRAME_OUT(); memset(byref(self.fo), 0, sizeof(self.fo))
        return True

    def frame(self):
        if self.cam.MV_CC_GetImageBuffer(self.fo, 1000) != 0:
            return None
        try:
            inf = self.fo.stFrameInfo; w, h = inf.nWidth, inf.nHeight
            tam = w*h*3
            if self.buf is None or self.buflen < tam:
                self.buf = (c_ubyte*tam)(); self.buflen = tam
            cv = MV_CC_PIXEL_CONVERT_PARAM(); memset(byref(cv),0,sizeof(cv))
            cv.nWidth=w; cv.nHeight=h; cv.pSrcData=self.fo.pBufAddr
            cv.nSrcDataLen=inf.nFrameLen; cv.enSrcPixelType=inf.enPixelType
            cv.enDstPixelType=PixelType_Gvsp_BGR8_Packed
            cv.pDstBuffer=self.buf; cv.nDstBufferSize=tam
            if self.cam.MV_CC_ConvertPixelType(cv) != 0:
                return None
            return np.frombuffer(ctypes.string_at(self.buf,tam),np.uint8).reshape((h,w,3)).copy()
        finally:
            self.cam.MV_CC_FreeImageBuffer(self.fo)

    def fechar(self):
        if self.cam:
            self.cam.MV_CC_StopGrabbing(); self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle(); MvCamera.MV_CC_Finalize()


def obter_frame_com_grelha(K, dist):
    """Loop ao vivo: corrige a lente, deteta a grelha, mostra feedback.
    Devolve (frame_corrigido, cols, linhas, corners) ao carregar ESPACO."""
    cam = CamLive()
    if not cam.abrir():
        print("[ERRO] Nao abriu a camara."); return (None,)*4

    print("Posiciona o tabuleiro no tapete. ESPACO quando estiver DETETADO. Q=sair.")
    resultado = (None,)*4
    novo_K = None
    try:
        while True:
            frame = cam.frame()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if novo_K is None:
                novo_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
            corr = cv2.undistort(frame, K, dist, None, novo_K)

            gray = cv2.cvtColor(corr, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
            c, l, corners = detetar_qualquer_grelha(gray_eq)
            if c is None:
                c, l, corners = detetar_qualquer_grelha(gray)

            disp = cv2.resize(corr, (1024, int(1024*h/w)))
            if c is not None:
                # refinar e desenhar
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit)
                esc = 1024.0 / w
                cv2.drawChessboardCorners(disp, (c, l), corners*esc, True)
                cv2.putText(disp, f"DETETADO ({c}x{l}) - ESPACO p/ capturar",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(disp, "Tabuleiro nao detetado - ajusta posicao/luz",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Calibracao perspetiva (ao vivo)", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' ') and c is not None:
                resultado = (corr, c, l, corners)
                break
            if k == ord('q'):
                break
    finally:
        cam.fechar()
        cv2.destroyAllWindows()
    return resultado


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagem", default=None, help="Foto ja tirada (salta a camara).")
    args = ap.parse_args()

    K, dist = carregar_calib_lente()
    print(f"[OK] Calibracao da lente carregada de {CALIB_LENTE}")

    if args.imagem:
        frame = cv2.imread(args.imagem)
        if frame is None:
            print(f"[ERRO] Nao li {args.imagem}"); sys.exit(1)
        h, w = frame.shape[:2]
        novo_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        corr = cv2.undistort(frame, K, dist, None, novo_K)
        gray = cv2.cvtColor(corr, cv2.COLOR_BGR2GRAY)
        c, l, corners = detetar_qualquer_grelha(cv2.createCLAHE(2.0,(8,8)).apply(gray))
        if c is None:
            c, l, corners = detetar_qualquer_grelha(gray)
        if c is None:
            print("[ERRO] Tabuleiro nao detetado na imagem."); sys.exit(1)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit)
    else:
        corr, c, l, corners = obter_frame_com_grelha(K, dist)
        if corr is None:
            print("[SAIR] Sem captura."); sys.exit(0)

    print(f"[DETETADO] Grelha {c}x{l} cantos internos.")

    # pontos de destino: grelha regular top-down
    passo = TAMANHO_QUADRADO * PX_POR_MM
    margem = passo
    dst = []
    for r in range(l):
        for cc in range(c):
            dst.append([cc*passo + margem, r*passo + margem])
    dst = np.array(dst, np.float32)
    src = corners.reshape(-1, 2).astype(np.float32)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        print("[ERRO] Falha na homografia."); sys.exit(1)

    out_w = int((c-1)*passo + 2*margem)
    out_h = int((l-1)*passo + 2*margem)
    rect = cv2.warpPerspective(corr, H, (out_w, out_h))
    cv2.imwrite("preview_perspetiva.png", rect)

    np.savez(FICHEIRO_SAIDA,
             homografia=H, out_size=np.array([out_w, out_h]),
             px_por_mm=np.array([PX_POR_MM]),
             camera_matrix=K, dist_coeffs=dist,
             grelha=np.array([c, l]))

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"\n[OK] Homografia calculada ({inliers}/{len(src)} cantos usados).")
    print(f"[GUARDADO] {FICHEIRO_SAIDA}")
    print(f"[PREVIEW]  preview_perspetiva.png  (confirma: quadrados certos, visto de cima)")
    print(f"  Escala: {PX_POR_MM} px/mm -> imagem retificada {out_w}x{out_h}px")


if __name__ == "__main__":
    main()

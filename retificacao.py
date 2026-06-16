# -*- coding: utf-8 -*-
"""
============================================================================
 retificacao.py — Modulo partilhado de correcao geometrica
============================================================================
Aplica as DUAS correcoes a um frame, pela ordem certa:
  1. Distorcao da lente   (de calibracao_camera_fujinon35.npz)
  2. Perspetiva top-down  (de perspetiva_tapete.npz)

Resultado: uma asa tem a MESMA forma esteja no centro ou no canto do tapete.

USO no C2R / no capturador:
    from retificacao import Retificador
    ret = Retificador()          # carrega os .npz uma vez
    frame_corrigido = ret.aplicar(frame)   # a cada frame, ANTES de segmentar

Se os ficheiros de calibracao nao existirem, aplicar() devolve o frame
original e avisa uma vez (modo "sem correcao"), para o codigo nao rebentar.
============================================================================
"""
import cv2
import numpy as np
import os

CALIB_LENTE  = "calibracao_camera_fujinon35.npz"
CALIB_PERSP  = "perspetiva_tapete.npz"


class Retificador:
    def __init__(self, pasta="."):
        self.ok_lente = False
        self.ok_persp = False
        self._mapx = None
        self._mapy = None
        self._H = None
        self._out_size = None
        self.px_por_mm = None
        self._avisou = False

        cam_path = os.path.join(pasta, CALIB_LENTE)
        per_path = os.path.join(pasta, CALIB_PERSP)

        # --- Calibracao da lente ---
        if os.path.exists(cam_path):
            d = np.load(cam_path, allow_pickle=True)
            self.K    = d["camera_matrix"]
            self.dist = d["dist_coeffs"]
            self.ok_lente = True
            print(f"[Retificador] Lente carregada ({CALIB_LENTE}).")
        else:
            print(f"[Retificador] AVISO: {CALIB_LENTE} nao encontrado.")

        # --- Perspetiva ---
        if os.path.exists(per_path):
            d = np.load(per_path, allow_pickle=True)
            self._H        = d["homografia"]
            self._out_size = tuple(int(x) for x in d["out_size"])
            self.px_por_mm = float(d["px_por_mm"][0]) if "px_por_mm" in d else None
            # se a perspetiva guardou tambem a calib da lente, usa-a como fallback
            if not self.ok_lente and "camera_matrix" in d:
                self.K = d["camera_matrix"]; self.dist = d["dist_coeffs"]
                self.ok_lente = True
            self.ok_persp = True
            print(f"[Retificador] Perspetiva carregada ({CALIB_PERSP}), "
                  f"saida {self._out_size}.")
        else:
            print(f"[Retificador] AVISO: {CALIB_PERSP} nao encontrado.")

    @property
    def ativo(self):
        return self.ok_lente or self.ok_persp

    def _prep_maps(self, w, h):
        """Pre-calcula os mapas de undistort (so uma vez, p/ velocidade)."""
        novo_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, novo_K, (w, h), cv2.CV_16SC2)

    def aplicar(self, frame):
        """Aplica lente + perspetiva. Devolve o frame corrigido.
        Se nada estiver calibrado, devolve o frame original."""
        if not self.ativo:
            if not self._avisou:
                print("[Retificador] Sem calibracao -> frame inalterado.")
                self._avisou = True
            return frame

        out = frame

        # 1) lente
        if self.ok_lente:
            h, w = out.shape[:2]
            if self._mapx is None or self._mapx.shape[:2] != (h, w):
                self._prep_maps(w, h)
            out = cv2.remap(out, self._mapx, self._mapy, cv2.INTER_LINEAR)

        # 2) perspetiva
        if self.ok_persp:
            out = cv2.warpPerspective(out, self._H, self._out_size,
                                      flags=cv2.INTER_LINEAR)
        return out

    def ponto_para_mm(self, x_px, y_px):
        """Converte um ponto da imagem RETIFICADA (px) para mm no tapete.
        Util se quiseres enviar coordenadas em mm ao robo. Requer perspetiva."""
        if not self.px_por_mm:
            return None
        return (x_px / self.px_por_mm, y_px / self.px_por_mm)


# Teste rapido standalone
if __name__ == "__main__":
    import sys
    ret = Retificador()
    print(f"\nAtivo: {ret.ativo}  (lente={ret.ok_lente}, perspetiva={ret.ok_persp})")
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            corr = ret.aplicar(img)
            cv2.imwrite("teste_retificado.png", corr)
            print(f"Guardado teste_retificado.png  ({img.shape} -> {corr.shape})")

# -*- coding: utf-8 -*-
"""
============================================================================
 INTERFACE GRAFICA — Sistema de Visao das Asas
============================================================================
Interface PyQt para operar o sistema sem terminal nem teclas.
REQUER: pip install PyQt5
USO:    py interface_asas.py
============================================================================
"""
import sys
import os
import time
import numpy as np
import cv2

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
                             QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QFont

# Importa TODA a logica do programa principal (sem a executar, gracas ao
# if __name__ == "__main__" do C2R). O ficheiro tem de estar na mesma pasta.
import C2R_v15 as motor


# ===========================================================================
# THREAD DO MOTOR DE VISAO
# ===========================================================================
class MotorVisao(QThread):
    novo_frame      = pyqtSignal(np.ndarray)
    estado_camara   = pyqtSignal(bool)
    estado_robo     = pyqtSignal(bool)
    n_pecas         = pyqtSignal(int)
    n_total         = pyqtSignal(int)
    estado_calib    = pyqtSignal(object)
    estado_modelos  = pyqtSignal(object)  # None/0 = nao carregado; int = nr asas
    envio_robo      = pyqtSignal(dict)
    log             = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ativo = True
        self.asa_selecionada = None
        self.camara = None
        self.robot = None
        self.registo = None
        self.escala_mm_px = None
        self._asa_pedida = None
        self._pedir_calibracao = False

    def calibrar(self):
        self._pedir_calibracao = True

    def selecionar_asa(self, asa):
        self._asa_pedida = asa

    def parar(self):
        self.ativo = False

    def run(self):
        if not motor.carregar_todas_as_asas():
            self.log.emit("ERRO: nao foi possivel carregar modelos das asas.")
            self.estado_modelos.emit(0)   # falhou -> indicador vermelho
            return
        self.log.emit("Modelos carregados: " + ", ".join(motor._asas_disponiveis()))

        disponiveis = motor._asas_disponiveis()
        # Avisa a interface quantas asas foram carregadas (indicador verde)
        self.estado_modelos.emit(len(disponiveis))
        self.asa_selecionada = disponiveis[0] if disponiveis else None

        self.robot = motor.RobotTM5()
        self.robot.ligar()
        self.estado_robo.emit(self.robot.ligado)

        _envio_original = self.robot.enviar_dados_asa
        def _envio_com_log(tipo_asa, cx, cy, rz, n_asas, invertida):
            resultado = _envio_original(tipo_asa, cx, cy, rz, n_asas, invertida)
            self.envio_robo.emit({
                "tipo_asa": tipo_asa, "x": cy, "y": -cx, "rz": rz,
                "n_asas": n_asas, "invertida": invertida,
            })
            return resultado
        self.robot.enviar_dados_asa = _envio_com_log

        self.camara = motor.CameraHikrobot()
        if not self.camara.ligar() or not self.camara.iniciar_captura():
            self.estado_camara.emit(False)
            self.log.emit("ERRO: nao foi possivel ligar a camara.")
            return
        self.estado_camara.emit(True)

        self.registo = motor.RegistoPecas(self.robot)
        ultimo_processamento = 0.0

        mtx, dist = motor.carregar_calibracao()
        self.escala_mm_px = None
        self.estado_calib.emit(self.escala_mm_px)
        self._mtx, self._dist = mtx, dist
        retificador = motor.Retificador() if motor.USAR_RETIFICACAO else None

        while self.ativo:
            frame = self.camara.capturar_frame_direto()
            if frame is None:
                time.sleep(0.01)
                continue

            if self._asa_pedida is not None:
                self.asa_selecionada = self._asa_pedida
                self._asa_pedida = None
                self.registo = motor.RegistoPecas(self.robot)

            if self._pedir_calibracao:
                self._pedir_calibracao = False
                self.log.emit("A iniciar calibracao de escala...")
                try:
                    nova_escala = motor.calibrar_escala(self.camara, self._mtx, self._dist)
                    if nova_escala is not None:
                        self.escala_mm_px = nova_escala
                        self.log.emit(f"Calibracao concluida: {nova_escala:.4f} mm/px")
                    self.estado_calib.emit(self.escala_mm_px)
                except Exception as e:
                    self.log.emit(f"Erro na calibracao: {e}")

            agora = time.time()
            frame_final = retificador.aplicar(frame) if retificador is not None else frame
            if not frame_final.flags.writeable:
                frame_final = frame_final.copy()

            if agora - ultimo_processamento >= motor.INTERVALO_PROCESSAMENTO:
                ultimo_processamento = agora
                self.registo.limpar_expiradas()

                contornos, _ = motor.extrair_contornos(frame_final)
                for cnt in contornos:
                    nome, score = motor.identificar_asa(cnt, self.asa_selecionada)
                    rect = cv2.minAreaRect(cnt)
                    box = np.intp(cv2.boxPoints(rect))
                    centro_px = (int(rect[0][0]), int(rect[0][1]))
                    centro_mm = motor.pixels_para_mm(centro_px, self.escala_mm_px)

                    ang_ponta = None
                    if not str(nome).startswith("Asa_0"):
                        ang_ponta = motor.angulo_pela_ponta(cnt)
                    if ang_ponta is not None:
                        angulo = ang_ponta
                    else:
                        ar = rect[2]; wr, hr = rect[1]
                        angulo = (ar + 90 if wr < hr else (ar + 180 if ar < 0 else ar))

                    self.registo.atualizar(centro_px, nome, centro_mm, angulo,
                                           score, box, self.asa_selecionada)

                self.registo.enviar_estado_robot(self.asa_selecionada)
                self.n_pecas.emit(len(self.registo.pecas))
                self.n_total.emit(self.registo.total_pecas_historico)
                self.estado_robo.emit(bool(self.robot.ligado))

            cor_asa = motor.CORES_PECAS.get(self.asa_selecionada, (0, 255, 0))
            for pid, peca in self.registo.pecas.items():
                centro = peca["centro_px"]; angulo_p = peca["angulo"]
                pos_mm = peca["centro_coord"]
                cor_estado = (0, 165, 255) if "Invertida" in peca["nome"] else cor_asa
                cv2.drawContours(frame_final, [peca["box"]], 0, cor_estado, 4)
                cv2.circle(frame_final, centro, 10, cor_estado, -1)
                rad = np.deg2rad(angulo_p)
                pt2 = (int(centro[0] + 100*np.cos(rad)), int(centro[1] + 100*np.sin(rad)))
                cv2.line(frame_final, centro, pt2, (0, 0, 255), 4)
                cv2.putText(frame_final, f"#{pid:04d} {peca['nome']}",
                            (centro[0]-80, centro[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_estado, 3)
                cv2.putText(frame_final, f"X:{pos_mm[0]:.1f} Y:{pos_mm[1]:.1f}",
                            (centro[0]-80, centro[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(frame_final, f"Rot: {int(angulo_p)} graus",
                            (centro[0]-80, centro[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            self.novo_frame.emit(frame_final)

        try:
            if self.robot: self.robot.desligar()
            if self.camara: self.camara.parar()
        except Exception:
            pass


# ===========================================================================
# WIDGET: indicador de estado
# ===========================================================================
class Indicador(QWidget):
    def __init__(self, etiqueta, txt_ativo="Ligado", txt_inativo="Desligado"):
        super().__init__()
        self.ativo = False
        self.etiqueta = etiqueta
        self.txt_ativo = txt_ativo
        self.txt_inativo = txt_inativo
        self.texto_livre = None
        self.setMinimumHeight(34)

    def set_estado(self, ativo):
        self.ativo = bool(ativo)
        self.update()

    def set_valor(self, valor):
        if valor is None:
            self.ativo = False
            self.texto_livre = "Nao calibrado"
        else:
            self.ativo = True
            self.texto_livre = f"{valor:.4f} mm/px"
        self.update()

    def set_modelos(self, n):
        """n None/0 -> nao carregado (vermelho); int>0 -> nr de asas (verde)."""
        if not n:
            self.ativo = False
            self.texto_livre = "Nao carregados"
        else:
            self.ativo = True
            self.texto_livre = f"{n} asas carregadas"
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cor = QColor(40, 200, 90) if self.ativo else QColor(210, 60, 60)
        p.setBrush(cor)
        p.setPen(Qt.NoPen)
        d = 18
        cy = self.height()//2
        p.drawEllipse(4, cy - d//2, d, d)
        p.setBrush(QColor(cor.red(), cor.green(), cor.blue(), 60))
        p.drawEllipse(0, cy - (d+8)//2, d+8, d+8)
        p.setPen(QColor(230, 230, 235))
        f = QFont("Segoe UI", 11)
        p.setFont(f)
        estado_txt = self.texto_livre if self.texto_livre is not None else (self.txt_ativo if self.ativo else self.txt_inativo)
        p.drawText(34, 0, self.width()-34, self.height(),
                   Qt.AlignVCenter | Qt.AlignLeft, f"{self.etiqueta}: {estado_txt}")


# ===========================================================================
# JANELA PRINCIPAL
# ===========================================================================
class JanelaPrincipal(QWidget):
    # Logotipo a mostrar no cabecalho. Procura-o NA PASTA DESTE FICHEIRO
    # (nao na pasta de trabalho), para o encontrar mesmo que o programa seja
    # corrido a partir de outra pasta (ex: do VS Code). Se nao existir, o
    # cabecalho mostra so o texto, sem erro.
    _PASTA_DESTE_FICHEIRO = os.path.dirname(os.path.abspath(__file__))
    LOGO = os.path.join(_PASTA_DESTE_FICHEIRO, "logo_ua.png")
    LOGO_DEM = os.path.join(_PASTA_DESTE_FICHEIRO, "logo_dem.png")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Visao — Asas")
        self.resize(1280, 800)
        self._build_ui()
        self._start_motor()

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #1e1f26; color: #e6e6ea; }
            QLabel#titulo { font-size: 18px; font-weight: 600; color: #ffffff; }
            QLabel#subtitulo { font-size: 12px; color: #9aa0ad; }
            QFrame#painel { background-color: #25262f; border-radius: 12px; }
            QPushButton#asa {
                background-color: #2d2f3a; border: 2px solid #3a3d4a;
                border-radius: 10px; padding: 14px; font-size: 15px; font-weight: 600;
            }
            QPushButton#asa:hover { background-color: #353846; border-color: #4a4e5e; }
            QPushButton#asa:checked {
                background-color: #2563eb; border-color: #3b82f6; color: white;
            }
            QPushButton#sair {
                background-color: #3a2d2d; border: 2px solid #5a3a3a;
                border-radius: 10px; padding: 10px; font-size: 13px;
            }
            QPushButton#sair:hover { background-color: #4a3636; }
            QPushButton#calibrar {
                background-color: #2d3a3a; border: 2px solid #3a5a5a;
                border-radius: 8px; padding: 8px; font-size: 13px; margin-top: 6px;
            }
            QPushButton#calibrar:hover { background-color: #364a4a; }
            QPushButton#calibrar:disabled { background-color: #2a2d33; color: #6a6e7a; border-color: #3a3d44; }
            QLabel#video { background-color: #15161c; border-radius: 12px; }
            QLabel#valor { font-size: 22px; font-weight: 700; color: #ffffff; }
            QLabel#chave { font-size: 11px; color: #9aa0ad; }
            QLabel#logenvio {
                font-family: Consolas, monospace; font-size: 13px; color: #7dd3a8;
                background-color: #1a1b22; border-radius: 8px; padding: 10px;
            }
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ---------- COLUNA ESQUERDA: video ----------
        col_video = QVBoxLayout()

        # Cabecalho: LOGO (esquerda) + textos (direita)
        cab = QHBoxLayout()
        cab.setSpacing(14)

        self.lbl_logo = QLabel()
        if os.path.exists(self.LOGO):
            pix = QPixmap(self.LOGO)
            if not pix.isNull():
                # Escala o logo para ~52px de altura, mantendo a proporcao
                self.lbl_logo.setPixmap(pix.scaledToHeight(52, Qt.SmoothTransformation))
                print(f"[logo] Logotipo carregado: {self.LOGO}")
            else:
                print(f"[logo] AVISO: ficheiro existe mas nao e uma imagem valida: {self.LOGO}")
        else:
            print(f"[logo] AVISO: logotipo NAO encontrado em: {self.LOGO}")
        self.lbl_logo.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        cab.addWidget(self.lbl_logo, 0, Qt.AlignVCenter)

        self.lbl_logo_dem = QLabel()
        if os.path.exists(self.LOGO_DEM):
            pix_dem = QPixmap(self.LOGO_DEM)
            if not pix_dem.isNull():
                # Escala o logo para ~52px de altura, mantendo a proporcao
                self.lbl_logo_dem.setPixmap(pix_dem.scaledToHeight(52, Qt.SmoothTransformation))
                print(f"[logo] Logotipo DEM carregado: {self.LOGO_DEM}")
            else:
                print(f"[logo] AVISO: ficheiro existe mas nao e uma imagem valida: {self.LOGO_DEM}")
        else:
            print(f"[logo] AVISO: logotipo DEM NAO encontrado em: {self.LOGO_DEM}")
        self.lbl_logo_dem.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        cab.addWidget(self.lbl_logo_dem, 0, Qt.AlignVCenter)

        textos = QVBoxLayout()
        textos.setSpacing(2)
        t = QLabel("Monitorizacao do Tapete"); t.setObjectName("titulo")
        s = QLabel("Deteccao e identificacao de asas em tempo real"); s.setObjectName("subtitulo")
        textos.addWidget(t); textos.addWidget(s)
        cab.addLayout(textos)
        cab.addStretch()
        col_video.addLayout(cab)

        self.video = QLabel(); self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(820, 620)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setText("A iniciar camara...")
        col_video.addWidget(self.video, stretch=1)
        root.addLayout(col_video, stretch=3)

        # ---------- COLUNA DIREITA: controlos ----------
        col_lado = QVBoxLayout()
        col_lado.setSpacing(14)

        painel_estado = QFrame(); painel_estado.setObjectName("painel")
        le = QVBoxLayout(painel_estado)
        le.setContentsMargins(16, 16, 16, 16); le.setSpacing(10)
        tit_est = QLabel("Estado das Ligacoes"); tit_est.setObjectName("titulo")
        tit_est.setStyleSheet("font-size: 15px;")
        le.addWidget(tit_est)
        self.ind_camara = Indicador("Camara")
        self.ind_robo = Indicador("Robo")
        self.ind_modelos = Indicador("Modelos")
        self.ind_modelos.set_modelos(0)   # arranca como nao carregados
        self.ind_calib = Indicador("Calibracao")
        self.ind_calib.set_valor(None)
        le.addWidget(self.ind_camara)
        le.addWidget(self.ind_robo)
        le.addWidget(self.ind_modelos)
        le.addWidget(self.ind_calib)

        self.btn_calibrar = QPushButton("Calibrar Escala")
        self.btn_calibrar.setObjectName("calibrar")
        self.btn_calibrar.clicked.connect(self._calibrar)
        le.addWidget(self.btn_calibrar)
        col_lado.addWidget(painel_estado)

        painel_met = QFrame(); painel_met.setObjectName("painel")
        lm = QGridLayout(painel_met)
        lm.setContentsMargins(16, 16, 16, 16)
        lm.setVerticalSpacing(4); lm.setHorizontalSpacing(20)
        k1 = QLabel("Asa selecionada"); k1.setObjectName("chave")
        self.lbl_asa = QLabel("—"); self.lbl_asa.setObjectName("valor")
        k2 = QLabel("Pecas no tapete"); k2.setObjectName("chave")
        self.lbl_npecas = QLabel("0"); self.lbl_npecas.setObjectName("valor")
        k3 = QLabel("Total detetadas"); k3.setObjectName("chave")
        self.lbl_total = QLabel("0"); self.lbl_total.setObjectName("valor")
        lm.addWidget(k1, 0, 0); lm.addWidget(k2, 0, 1)
        lm.addWidget(self.lbl_asa, 1, 0); lm.addWidget(self.lbl_npecas, 1, 1)
        lm.addWidget(k3, 2, 0)
        lm.addWidget(self.lbl_total, 3, 0)
        col_lado.addWidget(painel_met)

        painel_log = QFrame(); painel_log.setObjectName("painel")
        ll = QVBoxLayout(painel_log)
        ll.setContentsMargins(16, 16, 16, 16); ll.setSpacing(6)
        tit_log = QLabel("Ultimo Envio ao Robo"); tit_log.setObjectName("titulo")
        tit_log.setStyleSheet("font-size: 15px;")
        ll.addWidget(tit_log)
        self.lbl_envio = QLabel("Ainda nada enviado.")
        self.lbl_envio.setObjectName("logenvio")
        self.lbl_envio.setWordWrap(True)
        ll.addWidget(self.lbl_envio)
        col_lado.addWidget(painel_log)

        painel_asa = QFrame(); painel_asa.setObjectName("painel")
        la = QVBoxLayout(painel_asa)
        la.setContentsMargins(16, 16, 16, 16); la.setSpacing(10)
        tit_asa = QLabel("Escolher Asa a Identificar"); tit_asa.setObjectName("titulo")
        tit_asa.setStyleSheet("font-size: 15px;")
        la.addWidget(tit_asa)

        grid = QGridLayout(); grid.setSpacing(10)
        self.botoes_asa = {}
        asas = ["Asa_0", "Asa_1", "Asa_2", "Asa_3"]
        for i, asa in enumerate(asas):
            b = QPushButton(asa.replace("_", " "))
            b.setObjectName("asa")
            b.setCheckable(True)
            b.setMinimumHeight(56)
            b.clicked.connect(lambda _, a=asa: self._escolher_asa(a))
            self.botoes_asa[asa] = b
            grid.addWidget(b, i // 2, i % 2)
        la.addLayout(grid)
        col_lado.addWidget(painel_asa)

        col_lado.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        btn_sair = QPushButton("Sair"); btn_sair.setObjectName("sair")
        btn_sair.clicked.connect(self.close)
        col_lado.addWidget(btn_sair)

        root.addLayout(col_lado, stretch=1)

    def _start_motor(self):
        self.motor = MotorVisao()
        self.motor.novo_frame.connect(self._mostrar_frame)
        self.motor.estado_camara.connect(self.ind_camara.set_estado)
        self.motor.estado_robo.connect(self.ind_robo.set_estado)
        self.motor.n_pecas.connect(lambda n: self.lbl_npecas.setText(str(n)))
        self.motor.n_total.connect(lambda n: self.lbl_total.setText(str(n)))
        self.motor.estado_calib.connect(self.ind_calib.set_valor)
        self.motor.estado_modelos.connect(self.ind_modelos.set_modelos)
        self.motor.envio_robo.connect(self._mostrar_envio)
        self.motor.log.connect(lambda m: print("[motor]", m))
        self.motor.start()

    def _mostrar_envio(self, d):
        face = "Invertida" if d.get("invertida") else "Normal"
        texto = (f"X = {d['x']:.1f}    Y = {d['y']:.1f}\n"
                 f"Rotacao = {d['rz']:.1f}\n"
                 f"Asa {d['tipo_asa']} ({face})    Nr asas = {d['n_asas']}")
        self.lbl_envio.setText(texto)

    def _calibrar(self):
        if hasattr(self, "motor"):
            self.ind_calib.texto_livre = "A calibrar..."
            self.ind_calib.ativo = False
            self.ind_calib.update()
            self.motor.calibrar()

    def _escolher_asa(self, asa):
        for nome, b in self.botoes_asa.items():
            b.setChecked(nome == asa)
        self.lbl_asa.setText(asa.replace("_", " "))
        if hasattr(self, "motor"):
            self.motor.selecionar_asa(asa)

    def _mostrar_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video.width(), self.video.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def closeEvent(self, ev):
        try:
            if hasattr(self, "motor"):
                self.motor.parar()
                self.motor.wait(2000)
        except Exception:
            pass
        ev.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = JanelaPrincipal()
    janela.show()
    sys.exit(app.exec_())
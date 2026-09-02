"""TR: PySide6 acik/koyu tema ve ortak stil kurallari.
EN: PySide6 light/dark themes and shared style rules.
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from .qt_gui_common import configure_application_font, resource_path


def _dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#0b0b0b"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#f1f1f1"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#121212"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#1b1b1b"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#141414"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#f1f1f1"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#f1f1f1"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#202020"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#f1f1f1"))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#a8d6c5"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#080808"))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#c1eadb"))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#898989"))
    return palette


def _blue_palette() -> QtGui.QPalette:
    """The bright application palette is intentionally blue, never white."""
    palette = QtGui.QPalette()
    blue = QtGui.QColor("#0b4777")
    text = QtGui.QColor("#f4fbff")
    palette.setColor(QtGui.QPalette.ColorRole.Window, blue)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#0a3b63"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#135889"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#062844"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Text, text)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#12649f"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#66cef5"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#06243d"))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#b6eaff"))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#b8d4e5"))
    return palette


def _white_palette() -> QtGui.QPalette:
    """Neutral light palette kept as a separate user-selectable theme."""
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f4f7fb"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#172235"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#f2f7fb"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#112a42"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#15283e"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#1a314b"))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#102f4e"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#327aaf"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#176d9d"))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#718096"))
    return palette


def stylesheet(dark: bool, white: bool = False) -> str:
    """Return the shared, high-contrast presentation-oriented design system."""
    up_arrow = resource_path("assets/spin_up.svg").as_posix()
    down_arrow = resource_path("assets/spin_down.svg").as_posix()
    light_up_arrow = resource_path("assets/spin_up_dark.svg").as_posix()
    light_down_arrow = resource_path("assets/spin_down_dark.svg").as_posix()
    if dark:
        css = """
        QWidget { color: #edf3fb; font-size: 13px; }
        QMainWindow { background: #111824; }
        QWidget#previewPane, QWidget#comparisonPage { background: #111824; }
        QLabel#appTitle { color: #f7fbff; letter-spacing: 0.4px; }
        QLabel#appSubtitle { color: #aebed2; }
        QScrollArea#controlScroll { background: #151e2b; border: 1px solid #2b3a4d; border-radius: 14px; }
        QScrollArea#controlScroll > QWidget > QWidget { background: #151e2b; }
        QGroupBox#controlCard { background: #1a2534; border: 1px solid #2d4056; border-radius: 12px; margin-top: 14px; padding: 15px 10px 10px 10px; font-weight: 600; }
        QGroupBox#controlCard::title { subcontrol-origin: margin; left: 13px; padding: 0 6px; color: #dcecff; }
        QPushButton { color: #edf3fb; background: #223146; border: 1px solid #405773; border-radius: 8px; padding: 8px 12px; min-height: 18px; }
        QPushButton:hover { background: #2d4664; border-color: #7cc4ff; }
        QPushButton:pressed { background: #1a2a3d; }
        QPushButton[primary="true"] { color: #061420; background: #6fd0e8; border-color: #9ee9f6; font-weight: 700; }
        QPushButton[primary="true"]:hover { background: #9ae3f1; }
        QPushButton:disabled { color: #718096; background: #1b2430; border-color: #293747; }
        QToolButton { color: #d9e8f7; border: 1px solid #314961; border-radius: 8px; padding: 8px; background: #1c2a3a; font-weight: 600; }
        QToolButton:hover { border-color: #7cc4ff; background: #263b52; }
        QLineEdit, QSpinBox, QDoubleSpinBox { color: #f3f8ff; background: #101823; border: 1px solid #3a536f; border-radius: 7px; padding: 6px 8px; min-height: 20px; selection-background-color: #227ca6; }
        QComboBox { color: #f3f8ff; background: #101823; border: 1px solid #3a536f; border-radius: 7px; padding: 6px 30px 6px 8px; min-height: 20px; selection-background-color: #227ca6; }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #76ccff; padding: 5px 7px; }
        QComboBox:focus { border: 2px solid #76ccff; padding: 5px 29px 5px 7px; }
        QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 24px; border-left: 1px solid #3a536f; }
        QComboBox::down-arrow { image: url(__DOWN_ARROW__); width: 14px; height: 14px; }
        QCheckBox { spacing: 8px; color: #dce9f7; }
        QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #54718f; border-radius: 5px; background: #101823; }
        QCheckBox::indicator:checked { background: #60c7df; border-color: #9ce9f5; }
        QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 21px; background: #283d56; border-left: 1px solid #4f6e8e; border-bottom: 1px solid #4f6e8e; border-top-right-radius: 6px; }
        QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 21px; background: #283d56; border-left: 1px solid #4f6e8e; border-bottom-right-radius: 6px; }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #3c5978; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(__UP_ARROW__); width: 14px; height: 14px; }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(__DOWN_ARROW__); width: 14px; height: 14px; }
        QComboBox QAbstractItemView { color: #f3f8ff; background: #172233; selection-background-color: #285f87; border: 1px solid #4e6b8a; border-radius: 6px; outline: 0; padding: 0; }
        QComboBox QAbstractItemView::item { min-height: 30px; padding: 0 8px; margin: 0; border: 0; border-radius: 4px; }
        QComboBox QAbstractItemView::item:hover { background: #223f5b; }
        QComboBox QAbstractItemView QScrollBar:vertical { width: 12px; margin: 0; border: 0; background: #172233; }
        QComboBox QAbstractItemView QScrollBar::sub-line:vertical, QComboBox QAbstractItemView QScrollBar::add-line:vertical { height: 0; }
        QComboBox QAbstractItemView QScrollBar::handle:vertical { min-height: 24px; margin: 1px; border-radius: 5px; }
        QTabWidget::pane { background: #151f2e; border: 1px solid #2c4057; border-radius: 12px; top: -1px; }
        QTabBar::tab { color: #aebed2; background: transparent; padding: 10px 15px; margin: 0 3px 0 0; border-bottom: 3px solid transparent; }
        QTabBar::tab:hover { color: #f4f9ff; background: #1e2c3d; }
        QTabBar::tab:selected { color: #f7fbff; border-bottom-color: #6fd0e8; font-weight: 700; }
        QGraphicsView { border: 1px solid #2d4259; border-radius: 10px; background: #0d1420; }
        QFrame#metricCard { background: #19283a; border: 1px solid #36516d; border-radius: 10px; }
        QLabel#metricTitle { color: #93abc4; font-size: 10px; font-weight: 700; }
        QLabel#metricValue { color: #f3fbff; font-size: 18px; font-weight: 700; }
        QLabel#metricsDetails { color: #aebed2; background: #162334; border-left: 3px solid #6fd0e8; border-radius: 7px; padding: 7px 10px; }
        QTableWidget { gridline-color: #2d4056; alternate-background-color: #1b2737; selection-background-color: #285f87; }
        QHeaderView::section { background: #24354a; padding: 7px; border: 0; }
        QProgressBar { border: 1px solid #3f5a76; border-radius: 6px; text-align: center; background: #172233; }
        QProgressBar::chunk { background: #60c7df; border-radius: 5px; }
        QToolTip { color: #f7fbff; background: #09111c; border: 1px solid #76ccff; border-radius: 6px; padding: 6px; }
        QSplitter::handle { background: #2a3b50; width: 2px; }
        /* Scrollbar buttons and the draggable thumb use non-overlapping areas. */
        QScrollBar:vertical { background: #101823; width: 18px; margin: 18px 0 18px 0; border-left: 1px solid #31465d; }
        QScrollBar::handle:vertical { background: #49647e; border: 1px solid #6484a5; border-radius: 6px; min-height: 34px; margin: 2px; }
        QScrollBar::handle:vertical:hover { background: #5e84aa; }
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical { height: 18px; background: #24364c; border: 1px solid #496883; }
        QScrollBar::sub-line:vertical { subcontrol-origin: margin; subcontrol-position: top; border-top: 0; }
        QScrollBar::add-line:vertical { subcontrol-origin: margin; subcontrol-position: bottom; border-bottom: 0; }
        QScrollBar::sub-line:vertical:hover, QScrollBar::add-line:vertical:hover { background: #35536f; }
        QScrollBar::up-arrow:vertical { image: url(__UP_ARROW__); width: 14px; height: 14px; }
        QScrollBar::down-arrow:vertical { image: url(__DOWN_ARROW__); width: 14px; height: 14px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        QScrollBar:horizontal { background: #101823; height: 18px; margin: 0 18px 0 18px; border-top: 1px solid #31465d; }
        QScrollBar::handle:horizontal { background: #49647e; border: 1px solid #6484a5; border-radius: 6px; min-width: 34px; margin: 2px; }
        QScrollBar::handle:horizontal:hover { background: #5e84aa; }
        QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal { width: 18px; background: #24364c; border: 1px solid #496883; }
        QScrollBar::sub-line:horizontal { subcontrol-origin: margin; subcontrol-position: left; border-left: 0; }
        QScrollBar::add-line:horizontal { subcontrol-origin: margin; subcontrol-position: right; border-right: 0; }
        QScrollBar::sub-line:horizontal:hover, QScrollBar::add-line:horizontal:hover { background: #35536f; }
        QScrollBar::left-arrow:horizontal { width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-right: 7px solid #dbeeff; }
        QScrollBar::right-arrow:horizontal { width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 7px solid #dbeeff; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
        /* Dark is deliberately neutral black, not blue. */
        QMainWindow, QWidget#previewPane, QWidget#comparisonPage { background: #090909; }
        QScrollArea#controlScroll, QScrollArea#controlScroll > QWidget > QWidget { background: #111111; border-color: #303030; }
        QGroupBox#controlCard { background: #171717; border-color: #353535; }
        QGroupBox#controlCard::title { color: #f0f0f0; }
        QPushButton { background: #222222; border-color: #4a4a4a; color: #f1f1f1; }
        QPushButton:hover { background: #303030; border-color: #b9b9b9; }
        QPushButton[primary="true"] { background: #d4d4d4; border-color: #f3f3f3; color: #0c0c0c; }
        QToolButton, QToolButton#scrollStepButton { background: #202020; border-color: #505050; color: #f5f5f5; }
        QToolButton#scrollStepButton { min-width: 28px; min-height: 28px; font-size: 15px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #101010; border-color: #484848; color: #f3f3f3; }
        QComboBox::drop-down { border-left-color: #484848; }
        QComboBox QAbstractItemView { color: #f3f3f3; background: #151515; selection-background-color: #3d3d3d; border-color: #5a5a5a; }
        QComboBox QAbstractItemView::item:hover { background: #292929; }
        QTabWidget::pane { background: #111111; border-color: #303030; }
        QTabBar::tab:hover { background: #222222; } QTabBar::tab:selected { border-bottom-color: #d4d4d4; }
        QFrame#metricCard { background: #181818; border-color: #414141; }
        QLabel#metricTitle { color: #ababab; } QLabel#metricValue { color: #ffffff; }
        QLabel#metricsDetails { background: #161616; border-left-color: #d4d4d4; color: #d7d7d7; }
        QGraphicsView { background: #0b0b0b; border-color: #3b3b3b; }
        QScrollBar:vertical, QScrollBar:horizontal { background: #101010; border-color: #383838; }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #5a5a5a; border-color: #858585; }
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical, QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal { background: #222222; border-color: #505050; }
        """
        return css.replace("__UP_ARROW__", up_arrow).replace("__DOWN_ARROW__", down_arrow)
    css = """
        QWidget { color: #172235; font-size: 13px; }
        QMainWindow { background: #f4f7fb; }
        QWidget#previewPane, QWidget#comparisonPage { background: #f4f7fb; }
        QLabel#appTitle { color: #10233b; letter-spacing: 0.4px; }
        QLabel#appSubtitle { color: #52677f; }
        QScrollArea#controlScroll { background: #edf3f8; border: 1px solid #d4dfeb; border-radius: 14px; }
        QScrollArea#controlScroll > QWidget > QWidget { background: #edf3f8; }
        QGroupBox#controlCard { background: #ffffff; border: 1px solid #d5e0eb; border-radius: 12px; margin-top: 14px; padding: 15px 10px 10px 10px; font-weight: 600; }
        QGroupBox#controlCard::title { subcontrol-origin: margin; left: 13px; padding: 0 6px; color: #213a56; }
        QPushButton { color: #1a314b; border: 1px solid #bdcddd; border-radius: 8px; padding: 8px 12px; min-height: 18px; background: #ffffff; }
        QPushButton:hover { background: #edf7ff; border-color: #5e9bc9; }
        QPushButton:pressed { background: #dfeefa; }
        QPushButton[primary="true"] { color: #ffffff; background: #176d9d; border-color: #176d9d; font-weight: 700; }
        QPushButton[primary="true"]:hover { background: #0f82bc; }
        QPushButton:disabled { color: #93a2b1; background: #f4f6f8; border-color: #d8e0e8; }
        QToolButton { color: #24425f; border: 1px solid #c7d8e8; border-radius: 8px; padding: 8px; background: #f8fbff; font-weight: 600; }
        QToolButton:hover { border-color: #5e9bc9; background: #e9f5ff; }
        QLineEdit, QSpinBox, QDoubleSpinBox { color: #15283e; background: #ffffff; border: 1px solid #b9cadb; border-radius: 7px; padding: 6px 8px; min-height: 20px; selection-background-color: #327aaf; selection-color: #ffffff; }
        QComboBox { color: #15283e; background: #ffffff; border: 1px solid #b9cadb; border-radius: 7px; padding: 6px 30px 6px 8px; min-height: 20px; selection-background-color: #327aaf; selection-color: #ffffff; }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #2585c2; padding: 5px 7px; }
        QComboBox:focus { border: 2px solid #2585c2; padding: 5px 29px 5px 7px; }
        QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 24px; border-left: 1px solid #b9cadb; }
        QComboBox::down-arrow { image: url(__DOWN_ARROW__); width: 14px; height: 14px; }
        QCheckBox { spacing: 8px; color: #243a52; }
        QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #7d9ab6; border-radius: 5px; background: #ffffff; }
        QCheckBox::indicator:checked { background: #1883bb; border-color: #1883bb; }
        QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 21px; background: #eaf3fb; border-left: 1px solid #a6c0d8; border-bottom: 1px solid #a6c0d8; border-top-right-radius: 6px; }
        QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 21px; background: #eaf3fb; border-left: 1px solid #a6c0d8; border-bottom-right-radius: 6px; }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #d7ecfb; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(__UP_ARROW__); width: 14px; height: 14px; }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(__DOWN_ARROW__); width: 14px; height: 14px; }
        QComboBox QAbstractItemView { color: #15283e; background: #ffffff; selection-background-color: #cae6fa; border: 1px solid #a9bfd3; border-radius: 6px; outline: 0; padding: 0; }
        QComboBox QAbstractItemView::item { min-height: 30px; padding: 0 8px; margin: 0; border: 0; border-radius: 4px; }
        QComboBox QAbstractItemView::item:hover { background: #e5f3fd; }
        QComboBox QAbstractItemView QScrollBar:vertical { width: 12px; margin: 0; border: 0; background: #ffffff; }
        QComboBox QAbstractItemView QScrollBar::sub-line:vertical, QComboBox QAbstractItemView QScrollBar::add-line:vertical { height: 0; }
        QComboBox QAbstractItemView QScrollBar::handle:vertical { min-height: 24px; margin: 1px; border-radius: 5px; }
        QTabWidget::pane { background: #ffffff; border: 1px solid #d3dfeb; border-radius: 12px; top: -1px; }
        QTabBar::tab { color: #5a7088; background: transparent; padding: 10px 15px; margin: 0 3px 0 0; border-bottom: 3px solid transparent; }
        QTabBar::tab:hover { color: #173c5e; background: #e9f3fb; }
        QTabBar::tab:selected { color: #102f4e; border-bottom-color: #1685bd; font-weight: 700; }
        QGraphicsView { border: 1px solid #cbd9e7; border-radius: 10px; background: #f9fcff; }
        QFrame#metricCard { background: #ffffff; border: 1px solid #d3e1ee; border-radius: 10px; }
        QLabel#metricTitle { color: #668098; font-size: 10px; font-weight: 700; }
        QLabel#metricValue { color: #133c61; font-size: 18px; font-weight: 700; }
        QLabel#metricsDetails { color: #526b82; background: #eef7fd; border-left: 3px solid #1685bd; border-radius: 7px; padding: 7px 10px; }
        QTableWidget { color: #15283e; background: #ffffff; alternate-background-color: #f2f7fb; gridline-color: #d5e1eb; selection-background-color: #cae6fa; }
        QHeaderView::section { color: #24425f; background: #eaf2f9; padding: 7px; border: 0; }
        QProgressBar { border: 1px solid #b9ccdc; border-radius: 6px; text-align: center; background: #eef4f8; }
        QProgressBar::chunk { background: #1685bd; border-radius: 5px; }
        QToolTip { color: #ffffff; background: #112a42; border: 1px solid #4ba5df; border-radius: 6px; padding: 6px; }
        QSplitter::handle { background: #c6d6e5; width: 2px; }
        /* Separate arrow buttons from the draggable thumb in Blue as well. */
        QScrollBar:vertical { background: #e6eef6; width: 18px; margin: 18px 0 18px 0; border-left: 1px solid #bfd0df; }
        QScrollBar::handle:vertical { background: #8ca8c0; border: 1px solid #6689a7; border-radius: 6px; min-height: 34px; margin: 2px; }
        QScrollBar::handle:vertical:hover { background: #6d9cc1; }
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical { height: 18px; background: #d3e2ef; border: 1px solid #9bb9d0; }
        QScrollBar::sub-line:vertical { subcontrol-origin: margin; subcontrol-position: top; border-top: 0; }
        QScrollBar::add-line:vertical { subcontrol-origin: margin; subcontrol-position: bottom; border-bottom: 0; }
        QScrollBar::sub-line:vertical:hover, QScrollBar::add-line:vertical:hover { background: #bfdbef; }
        QScrollBar::up-arrow:vertical { image: url(__UP_ARROW__); width: 14px; height: 14px; }
        QScrollBar::down-arrow:vertical { image: url(__DOWN_ARROW__); width: 14px; height: 14px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        QScrollBar:horizontal { background: #e6eef6; height: 18px; margin: 0 18px 0 18px; border-top: 1px solid #bfd0df; }
        QScrollBar::handle:horizontal { background: #8ca8c0; border: 1px solid #6689a7; border-radius: 6px; min-width: 34px; margin: 2px; }
        QScrollBar::handle:horizontal:hover { background: #6d9cc1; }
        QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal { width: 18px; background: #d3e2ef; border: 1px solid #9bb9d0; }
        QScrollBar::sub-line:horizontal { subcontrol-origin: margin; subcontrol-position: left; border-left: 0; }
        QScrollBar::add-line:horizontal { subcontrol-origin: margin; subcontrol-position: right; border-right: 0; }
        QScrollBar::sub-line:horizontal:hover, QScrollBar::add-line:horizontal:hover { background: #bfdbef; }
        QScrollBar::left-arrow:horizontal { width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-right: 7px solid #244968; }
        QScrollBar::right-arrow:horizontal { width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 7px solid #244968; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
        """
    if white:
        return css.replace("__UP_ARROW__", light_up_arrow).replace("__DOWN_ARROW__", light_down_arrow)
    css += """
        /* Blue is a full blue workspace rather than the former white palette. */
        QWidget { color: #f2fbff; }
        QMainWindow, QWidget#previewPane, QWidget#comparisonPage { background: #0a4778; }
        QLabel#appTitle { color: #f4fbff; } QLabel#appSubtitle { color: #c1e4f7; }
        QScrollArea#controlScroll, QScrollArea#controlScroll > QWidget > QWidget { background: #0d548b; border-color: #4ba5d4; }
        QGroupBox#controlCard { background: #0f5d97; border-color: #56b1de; }
        QGroupBox#controlCard::title { color: #f1fbff; }
        QPushButton { background: #126aa8; border-color: #75c7ee; color: #f5fcff; }
        QPushButton:hover { background: #1a80c2; border-color: #b0e6ff; }
        QPushButton[primary="true"] { background: #8bdbff; border-color: #d2f3ff; color: #07375c; }
        QToolButton, QToolButton#scrollStepButton { background: #105f9b; border-color: #70c2eb; color: #ffffff; }
        QToolButton#scrollStepButton { min-width: 28px; min-height: 28px; font-size: 15px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #083c66; border-color: #71bde4; color: #f3fbff; selection-background-color: #86daf8; selection-color: #062d4e; }
        QComboBox::drop-down { border-left-color: #71bde4; }
        QComboBox QAbstractItemView { color: #f3fbff; background: #0a4c7c; selection-background-color: #2d8fc2; border-color: #75c7ee; }
        QComboBox QAbstractItemView::item:hover { background: #176fa6; }
        QCheckBox { color: #f2fbff; } QCheckBox::indicator { background: #07375e; border-color: #83cdf1; }
        QTabWidget::pane { background: #0d5286; border-color: #50a9d4; }
        QTabBar::tab { color: #c5e8fa; } QTabBar::tab:hover { background: #1674ad; color: #ffffff; } QTabBar::tab:selected { color: #ffffff; border-bottom-color: #93e1ff; }
        QFrame#metricCard { background: #0c548a; border-color: #57afd9; }
        QLabel#metricTitle { color: #b8e5fa; } QLabel#metricValue { color: #ffffff; }
        QLabel#metricsDetails { background: #0a4b7c; border-left-color: #8de3ff; color: #d5f2ff; }
        QGraphicsView { background: #07385f; border-color: #6bc3eb; }
        QTableWidget { background: #0b4c7d; color: #f2fbff; alternate-background-color: #105a91; gridline-color: #56aada; }
        QHeaderView::section { background: #12639c; color: #f2fbff; }
        QScrollBar:vertical, QScrollBar:horizontal { background: #083b63; border-color: #4aa4d4; }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #4caedb; border-color: #b4eaff; }
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical, QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal { background: #12669f; border-color: #75c7ee; }
        """
    return css.replace("__UP_ARROW__", light_up_arrow).replace("__DOWN_ARROW__", light_down_arrow)


def apply_theme(app: QtWidgets.QApplication, dark: bool, white: bool = False) -> None:
    """TR: Temayi tum uygulamaya uygular.
    EN: Apply the selected theme to the entire application.
    """
    app.setStyle("Fusion")
    configure_application_font(app)
    palette = _dark_palette() if dark else _white_palette() if white else _blue_palette()
    app.setPalette(palette)
    app.setStyleSheet(stylesheet(dark, white))

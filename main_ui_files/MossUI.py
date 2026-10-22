from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QPushButton
from modules.DragDropLineEdit import DragDropLineEdit
from ui_files.MossUI import Ui_moss_ui
from modules.BaseWidget import BaseWidget
from pathlib import Path


class MossWidget(BaseWidget):
    Toggled = Signal(bool)  # send to general args

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("MOSS-Image Args")
        self.widget = Ui_moss_ui()

        self.name = "moss_args"

        self.setup_widget()
        self.setup_connections()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

        selector_icon = QIcon(str(Path("icons/more-horizontal.svg")))
        # pipeline/component folders are the primary inputs; single-file paths can be typed in
        for elem, selector in (
            (self.widget.model_input, self.widget.model_selector),
            (self.widget.text_encoder_input, self.widget.text_encoder_selector),
            (self.widget.vae_input, self.widget.vae_selector),
        ):
            elem.setMode("folder")
            elem.highlight = True
            selector.setIcon(selector_icon)

    def setup_connections(self) -> None:
        self.widget.moss_training_box.clicked.connect(self.enable_disable)

        # File inputs — keys match sd_scripts argparse names
        self.widget.model_input.textChanged.connect(lambda x: self.edit_args("pretrained_model_name_or_path", x))
        self.widget.model_input.textChanged.connect(lambda _: self.change_vae())
        self.widget.model_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.model_input, "MOSS-Image Pipeline Folder")
        )
        self.widget.text_encoder_input.textChanged.connect(lambda x: self.edit_args("moss_text_encoder", x, True))
        self.widget.text_encoder_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.text_encoder_input, "Qwen3.5 Text Encoder Folder")
        )
        self.widget.vae_input.textChanged.connect(lambda _: self.change_vae())
        self.widget.vae_selector.clicked.connect(lambda: self.set_folder_from_dialog(self.widget.vae_input, "VAE Folder"))

        self.widget.max_token_input.valueChanged.connect(lambda x: self.edit_args("moss_max_token_length", x))
        self.widget.save_format_selector.currentTextChanged.connect(lambda x: self.edit_args("moss_lora_save_format", x))

        # Sampling Params
        self.widget.timestep_sampling_selector.currentIndexChanged.connect(self.change_timestep_sampling_type)
        self.widget.discrete_flow_shift_input.valueChanged.connect(lambda x: self.edit_args("discrete_flow_shift", x))
        self.widget.sigmoid_scale_input.valueChanged.connect(lambda x: self.edit_args("sigmoid_scale", x))
        self.widget.ts_mu_input.valueChanged.connect(lambda x: self.edit_args("moss_ts_mu", x))
        self.widget.ts_s_input.valueChanged.connect(lambda x: self.edit_args("moss_ts_s", x))
        self.widget.ts_cutoff_input.valueChanged.connect(lambda x: self.edit_args("moss_ts_cutoff", x))
        self.widget.ts_uniform_input.valueChanged.connect(lambda x: self.edit_args("moss_ts_uniform_mix", x))

        self.widget.blocks_to_swap_input.valueChanged.connect(self.change_blocks_to_swap)
        self.widget.fp32_modulation_enable.clicked.connect(lambda x: self.edit_args("moss_fp32_modulation", x, True))
        self.widget.attn_backend_selector.currentTextChanged.connect(lambda x: self.edit_args("moss_attn_backend", x))
        self.widget.compile_selector.currentTextChanged.connect(lambda x: self.edit_args("moss_compile", x))
        self.widget.int8_selector.currentIndexChanged.connect(self.change_int8)
        self.widget.sample_skip_layers_input.textChanged.connect(self.change_skip_layers)

    def enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.Toggled.emit(checked)
        if not checked:
            return

        self.edit_args("pretrained_model_name_or_path", self.widget.model_input.text())
        self.edit_args("moss_text_encoder", self.widget.text_encoder_input.text(), True)
        self.change_vae()
        self.edit_args("moss_max_token_length", self.widget.max_token_input.value())
        self.edit_args("moss_lora_save_format", self.widget.save_format_selector.currentText())
        self.change_timestep_sampling_type(self.widget.timestep_sampling_selector.currentIndex())
        self.change_blocks_to_swap(self.widget.blocks_to_swap_input.value())
        self.edit_args("moss_fp32_modulation", self.widget.fp32_modulation_enable.isChecked(), True)
        self.edit_args("moss_attn_backend", self.widget.attn_backend_selector.currentText())
        self.edit_args("moss_compile", self.widget.compile_selector.currentText())
        self.change_int8(self.widget.int8_selector.currentIndex())
        self.change_skip_layers(self.widget.sample_skip_layers_input.text())

    def change_int8(self, index: int) -> None:
        # 0 off, 1 on (bf16 backward), 2 on (int8 backward)
        self.edit_args("moss_int8", index > 0, True)
        if index == 2:
            self.edit_args("moss_int8_bwd", "int8")
        elif "moss_int8_bwd" in self.args:
            del self.args["moss_int8_bwd"]

    def change_vae(self) -> None:
        # always emit vae: sections merge in order, so an empty value here would let General Args' VAE
        # (e.g. an SDXL/Anima VAE) through. The pipeline folder itself is a valid VAE source.
        if not self.widget.moss_training_box.isChecked():
            return
        self.edit_args("vae", self.widget.vae_input.text().strip() or self.widget.model_input.text().strip(), True)

    def change_skip_layers(self, text: str) -> None:
        # validation drops empty strings (which would fall back to the default 22), so plain CFG is sent as "none"
        self.edit_args("moss_sample_uncond_skip_layers", text.strip() or "none")

    def external_enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.widget.moss_training_box.setEnabled(not checked)
        if self.widget.moss_training_box.isEnabled() and self.widget.moss_training_box.isChecked():
            self.enable_disable(True)

    def change_timestep_sampling_type(self, index: int) -> None:
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        self.edit_args("timestep_sampling", sampling_type)

        uses_moss = sampling_type == "moss"
        for name, spin in (
            ("moss_ts_mu", self.widget.ts_mu_input),
            ("moss_ts_s", self.widget.ts_s_input),
            ("moss_ts_cutoff", self.widget.ts_cutoff_input),
            ("moss_ts_uniform_mix", self.widget.ts_uniform_input),
        ):
            spin.setEnabled(uses_moss)
            if uses_moss:
                self.edit_args(name, spin.value())
            elif name in self.args:
                del self.args[name]

        uses_sigmoid_scale = sampling_type in ("sigmoid", "shift", "flux_shift")
        self.widget.sigmoid_scale_input.setEnabled(uses_sigmoid_scale)
        if uses_sigmoid_scale:
            self.edit_args("sigmoid_scale", self.widget.sigmoid_scale_input.value())
        elif "sigmoid_scale" in self.args:
            del self.args["sigmoid_scale"]

        uses_flow_shift = sampling_type in ("moss", "sigma", "shift")
        self.widget.discrete_flow_shift_input.setEnabled(uses_flow_shift)
        if uses_flow_shift:
            self.edit_args("discrete_flow_shift", self.widget.discrete_flow_shift_input.value())
        elif "discrete_flow_shift" in self.args:
            del self.args["discrete_flow_shift"]

    def change_blocks_to_swap(self, value: int) -> None:
        """Blocks to swap: 0 means disabled (None), any positive value is used as-is."""
        if value > 0:
            self.edit_args("blocks_to_swap", value)
        elif "blocks_to_swap" in self.args:
            del self.args["blocks_to_swap"]

    def load_args(self, args: dict) -> bool:
        args: dict = args.get(self.name, {})

        self.widget.moss_training_box.setChecked(bool(args))
        self.widget.model_input.setText(args.get("pretrained_model_name_or_path", ""))
        self.widget.text_encoder_input.setText(args.get("moss_text_encoder", ""))
        vae = args.get("vae", "")
        # a VAE equal to the model path is the auto-filled pipeline folder, shown as empty
        self.widget.vae_input.setText("" if vae == args.get("pretrained_model_name_or_path") else vae)
        self.widget.max_token_input.setValue(args.get("moss_max_token_length", 512))
        self.widget.save_format_selector.setCurrentText(args.get("moss_lora_save_format", "both"))

        self.widget.timestep_sampling_selector.setCurrentText(args.get("timestep_sampling", "moss"))
        self.widget.discrete_flow_shift_input.setValue(args.get("discrete_flow_shift", 1.0))
        self.widget.sigmoid_scale_input.setValue(args.get("sigmoid_scale", 1.0))
        self.widget.ts_mu_input.setValue(args.get("moss_ts_mu", -1.09))
        self.widget.ts_s_input.setValue(args.get("moss_ts_s", 0.95))
        self.widget.ts_cutoff_input.setValue(args.get("moss_ts_cutoff", 0.95))
        self.widget.ts_uniform_input.setValue(args.get("moss_ts_uniform_mix", 0.01))
        self.widget.blocks_to_swap_input.setValue(args.get("blocks_to_swap", 0))
        self.widget.fp32_modulation_enable.setChecked(args.get("moss_fp32_modulation", False))
        self.widget.attn_backend_selector.setCurrentText(args.get("moss_attn_backend", "auto"))
        self.widget.compile_selector.setCurrentText(args.get("moss_compile", "none"))
        int8_index = 0 if not args.get("moss_int8", False) else 2 if args.get("moss_int8_bwd") == "int8" else 1
        self.widget.int8_selector.setCurrentIndex(int8_index)
        skip_layers = str(args.get("moss_sample_uncond_skip_layers", "22"))
        self.widget.sample_skip_layers_input.setText("" if skip_layers == "none" else skip_layers)

        self.enable_disable(self.widget.moss_training_box.isChecked())

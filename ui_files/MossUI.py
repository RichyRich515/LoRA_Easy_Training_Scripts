# -*- coding: utf-8 -*-
# MOSS-Image args form (hand-written in the style of the generated AnimaUI.py)

from PySide6.QtCore import QCoreApplication, QMetaObject
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton

from modules.DragDropLineEdit import DragDropLineEdit
from modules.ScrollOnSelect import ComboBox, DoubleSpinBox, SpinBox


class Ui_moss_ui(object):
    def setupUi(self, moss_ui):
        if not moss_ui.objectName():
            moss_ui.setObjectName(u"moss_ui")
        moss_ui.resize(523, 420)
        self.gridLayout = QGridLayout(moss_ui)
        self.moss_training_box = QGroupBox(moss_ui)
        self.moss_training_box.setObjectName(u"moss_training_box")
        self.moss_training_box.setCheckable(True)
        self.moss_training_box.setChecked(False)
        self.gridLayout_2 = QGridLayout(self.moss_training_box)
        self.formLayout = QFormLayout()

        def path_row(row: int, name: str):
            label = QLabel(self.moss_training_box)
            self.formLayout.setWidget(row, QFormLayout.LabelRole, label)
            layout = QHBoxLayout()
            line_edit = DragDropLineEdit(self.moss_training_box)
            line_edit.setObjectName(f"{name}_input")
            layout.addWidget(line_edit)
            selector = QPushButton(self.moss_training_box)
            selector.setObjectName(f"{name}_selector")
            layout.addWidget(selector)
            self.formLayout.setLayout(row, QFormLayout.FieldRole, layout)
            return label, line_edit, selector

        # Model (maps to --pretrained_model_name_or_path), text encoder (--moss_text_encoder), VAE (--vae)
        self.label_model, self.model_input, self.model_selector = path_row(0, "model")
        self.label_te, self.text_encoder_input, self.text_encoder_selector = path_row(1, "text_encoder")
        self.label_vae, self.vae_input, self.vae_selector = path_row(2, "vae")

        # Tokens / LoRA export
        self.horizontalLayout_tokens = QHBoxLayout()
        self.label_tokens = QLabel(self.moss_training_box)
        self.horizontalLayout_tokens.addWidget(self.label_tokens)
        self.max_token_input = SpinBox(self.moss_training_box)
        self.max_token_input.setMinimum(32)
        self.max_token_input.setMaximum(4096)
        self.max_token_input.setValue(512)
        self.horizontalLayout_tokens.addWidget(self.max_token_input)
        self.label_save_format = QLabel(self.moss_training_box)
        self.horizontalLayout_tokens.addWidget(self.label_save_format)
        self.save_format_selector = ComboBox(self.moss_training_box)
        self.save_format_selector.addItem("both")
        self.save_format_selector.addItem("split")
        self.save_format_selector.addItem("fused")
        self.horizontalLayout_tokens.addWidget(self.save_format_selector)
        self.formLayout.setLayout(3, QFormLayout.SpanningRole, self.horizontalLayout_tokens)

        # Timestep sampling
        self.horizontalLayout_sampling = QHBoxLayout()
        self.label_sampling = QLabel(self.moss_training_box)
        self.horizontalLayout_sampling.addWidget(self.label_sampling)
        self.timestep_sampling_selector = ComboBox(self.moss_training_box)
        for item in ("moss", "sigmoid", "uniform", "sigma", "shift", "flux_shift"):
            self.timestep_sampling_selector.addItem(item)
        self.horizontalLayout_sampling.addWidget(self.timestep_sampling_selector)
        self.label_shift = QLabel(self.moss_training_box)
        self.horizontalLayout_sampling.addWidget(self.label_shift)
        self.discrete_flow_shift_input = DoubleSpinBox(self.moss_training_box)
        self.discrete_flow_shift_input.setDecimals(2)
        self.discrete_flow_shift_input.setSingleStep(0.1)
        self.discrete_flow_shift_input.setMinimum(0.01)
        self.discrete_flow_shift_input.setValue(1.0)
        self.horizontalLayout_sampling.addWidget(self.discrete_flow_shift_input)
        self.label_sigmoid = QLabel(self.moss_training_box)
        self.horizontalLayout_sampling.addWidget(self.label_sigmoid)
        self.sigmoid_scale_input = DoubleSpinBox(self.moss_training_box)
        self.sigmoid_scale_input.setDecimals(2)
        self.sigmoid_scale_input.setSingleStep(0.1)
        self.sigmoid_scale_input.setValue(1.0)
        self.horizontalLayout_sampling.addWidget(self.sigmoid_scale_input)
        self.formLayout.setLayout(4, QFormLayout.SpanningRole, self.horizontalLayout_sampling)

        # MOSS recipe sampler parameters
        self.horizontalLayout_moss_ts = QHBoxLayout()

        def ts_spin(label_attr: str, spin_attr: str, minimum: float, maximum: float, value: float):
            label = QLabel(self.moss_training_box)
            setattr(self, label_attr, label)
            self.horizontalLayout_moss_ts.addWidget(label)
            spin = DoubleSpinBox(self.moss_training_box)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            spin.setMinimum(minimum)
            spin.setMaximum(maximum)
            spin.setValue(value)
            setattr(self, spin_attr, spin)
            self.horizontalLayout_moss_ts.addWidget(spin)

        ts_spin("label_ts_mu", "ts_mu_input", -10.0, 10.0, -1.09)
        ts_spin("label_ts_s", "ts_s_input", 0.01, 10.0, 0.95)
        ts_spin("label_ts_cutoff", "ts_cutoff_input", 0.05, 1.0, 0.95)
        ts_spin("label_ts_uniform", "ts_uniform_input", 0.0, 1.0, 0.01)
        self.formLayout.setLayout(5, QFormLayout.SpanningRole, self.horizontalLayout_moss_ts)

        # Memory
        self.horizontalLayout_mem = QHBoxLayout()
        self.label_blocks_to_swap = QLabel(self.moss_training_box)
        self.horizontalLayout_mem.addWidget(self.label_blocks_to_swap)
        self.blocks_to_swap_input = SpinBox(self.moss_training_box)
        self.blocks_to_swap_input.setMinimum(0)
        self.blocks_to_swap_input.setMaximum(40)
        self.blocks_to_swap_input.setValue(0)
        self.horizontalLayout_mem.addWidget(self.blocks_to_swap_input)
        self.fp32_modulation_enable = QCheckBox(self.moss_training_box)
        self.horizontalLayout_mem.addWidget(self.fp32_modulation_enable)
        self.label_skip_layers = QLabel(self.moss_training_box)
        self.horizontalLayout_mem.addWidget(self.label_skip_layers)
        self.sample_skip_layers_input = QLineEdit(self.moss_training_box)
        self.sample_skip_layers_input.setText("22")
        self.sample_skip_layers_input.setMaximumWidth(80)
        self.horizontalLayout_mem.addWidget(self.sample_skip_layers_input)
        self.horizontalLayout_mem.addStretch()
        self.formLayout.setLayout(6, QFormLayout.SpanningRole, self.horizontalLayout_mem)

        # Speed
        self.horizontalLayout_speed = QHBoxLayout()
        self.label_attn = QLabel(self.moss_training_box)
        self.horizontalLayout_speed.addWidget(self.label_attn)
        self.attn_backend_selector = ComboBox(self.moss_training_box)
        for item in ("auto", "cudnn", "efficient", "flash", "torch"):
            self.attn_backend_selector.addItem(item)
        self.horizontalLayout_speed.addWidget(self.attn_backend_selector)
        self.label_compile = QLabel(self.moss_training_box)
        self.horizontalLayout_speed.addWidget(self.label_compile)
        self.compile_selector = ComboBox(self.moss_training_box)
        for item in ("none", "situ", "blocks"):
            self.compile_selector.addItem(item)
        self.horizontalLayout_speed.addWidget(self.compile_selector)
        self.label_int8 = QLabel(self.moss_training_box)
        self.horizontalLayout_speed.addWidget(self.label_int8)
        self.int8_selector = ComboBox(self.moss_training_box)
        for item in ("off", "on", "on (int8 backward)"):
            self.int8_selector.addItem(item)
        self.horizontalLayout_speed.addWidget(self.int8_selector)
        self.horizontalLayout_speed.addStretch()
        self.formLayout.setLayout(7, QFormLayout.SpanningRole, self.horizontalLayout_speed)

        self.gridLayout_2.addLayout(self.formLayout, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.moss_training_box, 0, 0, 1, 1)

        self.retranslateUi(moss_ui)
        QMetaObject.connectSlotsByName(moss_ui)

    def retranslateUi(self, moss_ui):
        tr = lambda text: QCoreApplication.translate("moss_ui", text, None)
        moss_ui.setWindowTitle(tr(u"Form"))
        self.moss_training_box.setTitle(tr(u"Train MOSS-Image"))
        self.label_model.setText(tr(u"MOSS-Image Model"))
        self.label_te.setText(tr(u"Text Encoder"))
        self.label_vae.setText(tr(u"VAE"))
        self.label_tokens.setText(tr(u"Max Tokens"))
        self.label_save_format.setText(tr(u"LoRA Key Format"))
        self.label_sampling.setText(tr(u"Timestep Sampling"))
        self.label_shift.setText(tr(u"Discrete Flow Shift"))
        self.label_sigmoid.setText(tr(u"Sigmoid Scale"))
        self.label_ts_mu.setText(tr(u"mu"))
        self.label_ts_s.setText(tr(u"s"))
        self.label_ts_cutoff.setText(tr(u"cutoff"))
        self.label_ts_uniform.setText(tr(u"uniform mix"))
        self.label_blocks_to_swap.setText(tr(u"Blocks to Swap"))
        self.fp32_modulation_enable.setText(tr(u"FP32 Modulation"))
        self.label_attn.setText(tr(u"Attention"))
        self.label_compile.setText(tr(u"Compile"))
        self.attn_backend_selector.setToolTip(tr(
            u"<html><head/><body><p>Attention kernel. <b>auto</b>: PyTorch's own choice (the memory-efficient kernel on Windows). "
            u"<b>cudnn</b> is much faster (~25% per step with Compile = blocks) but needs torch 2.14.1+cu130 (cuDNN 9.24): "
            u"with torch 2.7.1 (cuDNN 9.7.1) its backward produced NaN gradients on real training data. "
            u"Maps to <code>--moss_attn_backend</code>.</p></body></html>"))
        self.compile_selector.setToolTip(tr(
            u"<html><head/><body><p>torch.compile (requires Triton). <b>situ</b>: fuses the SiTU activation (~18% faster). "
            u"<b>blocks</b>: compiles each transformer block (~25% faster, ~16 s compile at start, less VRAM). "
            u"<b>none</b>: eager. With Blocks to Swap, blocks falls back to situ. "
            u"Maps to <code>--moss_compile</code>.</p></body></html>"))
        self.label_int8.setText(tr(u"INT8 Base"))
        self.int8_selector.setToolTip(tr(
            u"<html><head/><body><p>Quantize the frozen DiT block weights to ConvRot INT8 (the same recipe as the ComfyUI "
            u"<code>*_int8_convrot</code> checkpoints) and run them through cuBLAS int8. "
            u"Faster on GeForce cards with Compile = blocks (slower without it), ~4.5 GB less VRAM; the LoRA learns "
            u"against the int8 base. "
            u"<b>int8 backward</b> also quantizes the gradient pass (faster, noisier). Not compatible with Blocks to Swap. "
            u"Maps to <code>--moss_int8</code> / <code>--moss_int8_bwd</code>.</p></body></html>"))
        self.label_skip_layers.setText(tr(u"Sample CFG Skip Layers"))
        self.fp32_modulation_enable.setToolTip(tr(
            u"<html><head/><body><p>Keep the timestep MLP and modulation projections in fp32 (KohakuMossTrainer does this). "
            u"They are frozen; this only avoids bf16 rounding of the modulation. Negligible memory cost. "
            u"Maps to <code>--moss_fp32_modulation</code>.</p></body></html>"))
        self.sample_skip_layers_input.setToolTip(tr(
            u"<html><head/><body><p>Comma-separated block indices skipped in the negative CFG pass of sample images. "
            u"The official MOSS-Image pipeline uses 22. Empty for plain CFG. Only affects samples, not training. "
            u"Maps to <code>--moss_sample_uncond_skip_layers</code>.</p></body></html>"))

        self.model_input.setPlaceholderText(tr(u"MOSS-Image-Base pipeline folder (or a bf16 transformer .safetensors) [Required]"))
        self.model_input.setToolTip(tr(
            u"<html><head/><body><p>The OpenMOSS-Team/MOSS-Image-Base folder (with transformer/, text_encoder/, tokenizer/, vae/). "
            u"Base is recommended for LoRA training; LoRAs trained on Base also apply to SFT. "
            u"A single bf16 transformer file (split or fused qkv) also works if Text Encoder and VAE are set. "
            u"Quantized (int8/fp8) checkpoints cannot be trained. Maps to <code>--pretrained_model_name_or_path</code>.</p></body></html>"))
        self.text_encoder_input.setPlaceholderText(tr(u"Qwen3.5-9B folder (bf16, HF format) [Optional with a pipeline folder]"))
        self.text_encoder_input.setToolTip(tr(
            u"<html><head/><body><p>Qwen3.5-9B text encoder folder in HF format (bf16). Leave empty to use the pipeline folder's "
            u"text_encoder/. It runs in a separate worker before training (needs ~18 GB VRAM) and its outputs are cached; "
            u"transformers 5 is installed automatically into <code>sd_scripts/moss_te_deps</code> on first use. "
            u"ComfyUI int8/w4a8 text encoder files are not supported. Maps to <code>--moss_text_encoder</code>.</p></body></html>"))
        self.vae_input.setPlaceholderText(tr(u"VAE folder or flux2-small-decoder file [Optional with a pipeline folder]"))
        self.vae_input.setToolTip(tr(
            u"<html><head/><body><p>MOSS-Image VAE (FLUX.2 encoder + small decoder). Leave empty to use the pipeline folder's vae/ "
            u"(recommended: its BatchNorm statistics define the latent space). A single flux2-small-decoder .safetensors also works. "
            u"Maps to <code>--vae</code>.</p></body></html>"))
        self.max_token_input.setToolTip(tr(
            u"<html><head/><body><p>Maximum caption length in Qwen3.5 tokens, including the chat template. Default: 512 "
            u"(MOSS-Image supports up to 4096). Maps to <code>--moss_max_token_length</code>.</p></body></html>"))
        self.save_format_selector.setToolTip(tr(
            u"<html><head/><body><p>Attention LoRA keys written to the saved file."
            u"<br/><b>both</b> (recommended): split q/k/v keys and fused qkv keys. Each loader applies only the keys matching its model."
            u"<br/><b>split</b>: proj.q/k/v keys for Base/diffusers-layout models."
            u"<br/><b>fused</b>: proj.qkv keys for fused single-file checkpoints (e.g. moss_image_sft_bf16 in Forge). "
            u"Without fused keys, Forge silently ignores attention LoRAs on fused checkpoints."
            u"<br/>Maps to <code>--moss_lora_save_format</code>.</p></body></html>"))
        self.timestep_sampling_selector.setToolTip(tr(
            u"<html><head/><body><p>Timestep sampling."
            u"<br/><b>moss</b> (recommended): the MOSS-Image training recipe, sigma = 1 - sigmoid(mu + s&middot;N(0,1)) truncated at "
            u"<i>cutoff</i> and rescaled, mixed with uniform. Biased toward high noise."
            u"<br/><b>sigmoid</b>: logit-normal around sigma 0.5. <b>uniform</b>: flat. <b>sigma</b>/<b>shift</b>/<b>flux_shift</b>: "
            u"as in the other flow models.</p></body></html>"))
        self.discrete_flow_shift_input.setToolTip(tr(
            u"<html><head/><body><p>Timestep shift. MOSS-Image trains with 1.0 (its inference shift of 5.0 is applied only "
            u"when sampling). Used by <code>moss</code>, <code>sigma</code> and <code>shift</code>.</p></body></html>"))
        self.blocks_to_swap_input.setToolTip(tr(
            u"<html><head/><body><p>Number of the 42 transformer blocks to keep in CPU RAM and swap in during forward/backward. "
            u"Not needed on 24 GB GPUs at 1024px with gradient checkpointing. 0 disables.</p></body></html>"))

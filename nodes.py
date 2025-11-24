import os
import re
import torch
import s3tokenizer
import onnxruntime
import torchaudio
from typing import Dict, Any, List

from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.utils.text import normalize_text
from soulxpodcast.utils.audio import mel_spectrogram, audio_volume_normalize
from soulxpodcast.utils.commons import set_all_random_seed
from soulxpodcast.engine.llm_engine import HFLLMEngine
import torchaudio.compliance.kaldi as kaldi

SPK_DICT = [
    "<|SPEAKER_0|>", "<|SPEAKER_1|>", "<|SPEAKER_2|>", "<|SPEAKER_3|>",
    "<|SPEAKER_4|>", "<|SPEAKER_5|>", "<|SPEAKER_6|>", "<|SPEAKER_7|>",
    "<|SPEAKER_8|>", "<|SPEAKER_9|>"
]
MAX_SUPPORTED_SPEAKERS = len(SPK_DICT)  # Maximum number of speakers supported (10)
TEXT_START, TEXT_END, AUDIO_START = "<|text_start|>", "<|text_end|>", "<|semantic_token_start|>"
TASK_PODCAST = "<|task_podcast|>"



class SoulXPodcastLoader:
    
    @classmethod
    def get_tts_dir(cls):
        current_file = os.path.abspath(__file__)
        comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        tts_dir = os.path.join(comfy_root, "models", "TTS")
        return tts_dir
    
    @classmethod
    def get_model_list(cls):
        tts_dir = cls.get_tts_dir()
        
        model_list = []
        if os.path.exists(tts_dir) and os.path.isdir(tts_dir):
            for item in os.listdir(tts_dir):
                item_path = os.path.join(tts_dir, item)
                if os.path.isdir(item_path):
                    config_file = os.path.join(item_path, "soulxpodcast_config.json")
                    flow_file = os.path.join(item_path, "flow.pt")
                    hift_file = os.path.join(item_path, "hift.pt")
                    onnx_file = os.path.join(item_path, "campplus.onnx")
                    if os.path.exists(config_file) and os.path.exists(flow_file) and os.path.exists(hift_file) and os.path.exists(onnx_file):
                        model_list.append(item)
        
        model_list.sort()
        
        if not model_list:
            return ["No models found"]
        
        return model_list
    
    @classmethod
    def INPUT_TYPES(cls):
        model_list = cls.get_model_list()
        return {
            "required": {
                "model_name": (model_list, {
                    "default": model_list[0] if model_list else "No models found",
                }),
                "llm_engine": (["hf", "vllm"], {
                    "default": "hf"
                }),
                "fp16_flow": ("BOOLEAN", {
                    "default": False
                }),
                "seed": ("INT", {
                    "default": 1988,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("SOULX_MODEL",)
    FUNCTION = "load_models"
    CATEGORY = "SoulX-Podcast"
    
    def load_models(self, model_name: str, llm_engine: str = "hf", fp16_flow: bool = False, seed: int = 1988):
        set_all_random_seed(seed)
        
        tts_dir = SoulXPodcastLoader.get_tts_dir()
        model_path = os.path.join(tts_dir, model_name)
        
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
            initial_values={"fp16_flow": fp16_flow},
            json_file=f"{model_path}/soulxpodcast_config.json"
        )
        
        if llm_engine == "vllm":
            import importlib.util
            if not importlib.util.find_spec("vllm"):
                llm_engine = "hf"
                print(f"[WARNING]: VLLM 未安装，切换到 hf engine。")
        
        config = Config(
            model=model_path,
            enforce_eager=True,
            llm_engine=llm_engine,
            hf_config=hf_config
        )
        
        model = SoulXPodcast(config)
        
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        spk_model = onnxruntime.InferenceSession(
            f"{model_path}/campplus.onnx",
            sess_options=option,
            providers=["CPUExecutionProvider"]
        )
        
        soulx_model = {
            "model": model,
            "config": config,
            "spk_model": spk_model,
            "audio_tokenizer": model.audio_tokenizer,
            "llm": model.llm,
            "flow": model.flow,
            "hift": model.hift,
            "tokenizer": model.llm.tokenizer,
        }
        
        return (soulx_model,)


class SoulXPodcastInputParser:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soulx_model": ("SOULX_MODEL",),
                "input_mode": (["simple", "json"], {
                    "default": "simple",
                    "tooltip": "simple: Simple mode (multi-speaker dialogue, using node inputs)\njson: JSON mode (multi-speaker dialogue, using JSON config)"
                }),
            },
            "optional": {
                "S1_prompt_audio": ("AUDIO",),
                "S2_prompt_audio": ("AUDIO",),
                "S3_prompt_audio": ("AUDIO",),
                "S4_prompt_audio": ("AUDIO",),
                "S5_prompt_audio": ("AUDIO",),
                "S6_prompt_audio": ("AUDIO",),
                "S7_prompt_audio": ("AUDIO",),
                "S8_prompt_audio": ("AUDIO",),
                "S9_prompt_audio": ("AUDIO",),
                "S10_prompt_audio": ("AUDIO",),
                "dialogue_script": ("STRING", {
                    "multiline": True,
                    "default": "[S1] Hello there, Xiaoxi.\n[S2] Hello, Nenglao!",
                    "tooltip": "Dialogue script, format:\n[S1] First sentence\n[S2] Second sentence\n[S3] Third sentence\n...\nSupports inline pause tags: <|pause:MS|> where MS is milliseconds\nExample: [S1] Hello <|pause:500|> how are you?\nThe system will automatically extract the first sentence from each speaker as prompt text"
                }),
                "diff_spk_pause_ms": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5000,
                    "step": 50,
                    "tooltip": "Pause duration in milliseconds between different speakers"
                }),
                "json_config": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "JSON format config, supports multi-speaker dialogue (up to 10 speakers). Format:\n{\n  \"speakers\": {\n    \"S1\": {\"prompt_audio\": \"AUDIO input\", \"dialect_prompt\": \"<|Henan|>...\"},\n    \"S2\": {...},\n    ...\n  },\n  \"dialogue_script\": \"[S1]...\\n[S2]...\\n[S3]...\"\n}\nNote: prompt_audio needs to be connected to AUDIO input first and referenced by variable name"
                }),
            }
        }
    
    RETURN_TYPES = ("PODCAST_INPUT",)
    FUNCTION = "parse_input"
    CATEGORY = "SoulX-Podcast"
    
    @staticmethod
    def _extract_text_from_dialect(dialect_text: str) -> str:
        if not dialect_text:
            return ""
        dialect_tags = ["<|Henan|>", "<|Sichuan|>", "<|Yue|>"]
        text = dialect_text
        for tag in dialect_tags:
            if text.startswith(tag):
                text = text[len(tag):].strip()
                break
        return text
    
    def parse_input(
        self,
        soulx_model: Dict[str, Any],
        input_mode: str = "simple",
        S1_prompt_audio=None,
        S2_prompt_audio=None,
        S3_prompt_audio=None,
        S4_prompt_audio=None,
        S5_prompt_audio=None,
        S6_prompt_audio=None,
        S7_prompt_audio=None,
        S8_prompt_audio=None,
        S9_prompt_audio=None,
        S10_prompt_audio=None,
        dialogue_script: str = "",
        diff_spk_pause_ms = 0,
        json_config: str = "{}",
    ):
        # Handle diff_spk_pause_ms type conversion defensively
        # ComfyUI might pass unexpected values when optional parameters are not connected
        if diff_spk_pause_ms is None or diff_spk_pause_ms == "":
            diff_spk_pause_ms = 0
        elif isinstance(diff_spk_pause_ms, str):
            try:
                diff_spk_pause_ms = int(diff_spk_pause_ms)
            except (ValueError, TypeError):
                diff_spk_pause_ms = 0
        elif not isinstance(diff_spk_pause_ms, int):
            try:
                diff_spk_pause_ms = int(diff_spk_pause_ms)
            except (ValueError, TypeError):
                diff_spk_pause_ms = 0
        DEFAULT_PROMPTS = {
            "S1": "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
            "S2": "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
            "S3": "Speaker 3 default prompt text",
            "S4": "Speaker 4 default prompt text",
            "S5": "Speaker 5 default prompt text",
            "S6": "Speaker 6 default prompt text",
            "S7": "Speaker 7 default prompt text",
            "S8": "Speaker 8 default prompt text",
            "S9": "Speaker 9 default prompt text",
            "S10": "Speaker 10 default prompt text",
        }
        
        config = soulx_model["config"]
        tokenizer = soulx_model["tokenizer"]
        spk_model = soulx_model["spk_model"]
        
        if input_mode == "json":
            json_speakers_data, dialogue_script_from_json = self._parse_json_config(json_config)
            
            if dialogue_script_from_json:
                dialogue_script = dialogue_script_from_json
            elif not dialogue_script:
                raise ValueError(
                    "In JSON mode, dialogue_script must be provided in the 'dialogue_script' field of the JSON config, "
                    "or in the node's dialogue_script parameter.\n"
                    "JSON format example:\n"
                    '{\n  "speakers": {...},\n  "dialogue_script": "[S1] First sentence\\n[S2] Second sentence"\n}'
                )
            
            speakers_data = {}
            audio_inputs = {
                "S1": S1_prompt_audio,
                "S2": S2_prompt_audio,
                "S3": S3_prompt_audio,
                "S4": S4_prompt_audio,
                "S5": S5_prompt_audio,
                "S6": S6_prompt_audio,
                "S7": S7_prompt_audio,
                "S8": S8_prompt_audio,
                "S9": S9_prompt_audio,
                "S10": S10_prompt_audio,
            }
            
            parsed_script_texts = {}
            if dialogue_script:
                try:
                    temp_text_list, temp_spk_list = self._parse_dialogue_script(dialogue_script)
                    for idx, (text, spk_id) in enumerate(zip(temp_text_list, temp_spk_list)):
                        spk_key = f"S{spk_id + 1}"
                        if spk_key not in parsed_script_texts:
                            parsed_script_texts[spk_key] = text
                except Exception:
                    pass
            
            for spk_name in json_speakers_data:
                if audio_inputs.get(spk_name) is not None:
                    # 优先使用json中的prompt_text（如有）
                    prompt_text = json_speakers_data[spk_name].get("prompt_text", "")
                    # 没有就用默认，不再自动抽取
                    if not prompt_text:
                        prompt_text = DEFAULT_PROMPTS.get(spk_name, "")
                    if prompt_text:
                        speakers_data[spk_name] = {
                            "prompt_audio": audio_inputs[spk_name],
                            "prompt_text": prompt_text,
                            "dialect_prompt": json_speakers_data[spk_name].get("dialect_prompt", "")
                        }
                    else:
                        raise ValueError(
                            f"Audio for {spk_name} is provided, but cannot extract prompt text! Please set it or check config."
                        )
        else:
            # simple模式
            speakers_data = {}
            
            parsed_script_texts = {}
            if dialogue_script:
                try:
                    temp_text_list, temp_spk_list = self._parse_dialogue_script(dialogue_script)
                    for idx, (text, spk_id) in enumerate(zip(temp_text_list, temp_spk_list)):
                        spk_key = f"S{spk_id + 1}"
                        if spk_key not in parsed_script_texts:
                            parsed_script_texts[spk_key] = text
                except Exception:
                    pass
            
            # prompt_text 只用于音色克隆，不参与播客内容生成。
            audio_inputs_simple = {
                "S1": S1_prompt_audio,
                "S2": S2_prompt_audio,
                "S3": S3_prompt_audio,
                "S4": S4_prompt_audio,
                "S5": S5_prompt_audio,
                "S6": S6_prompt_audio,
                "S7": S7_prompt_audio,
                "S8": S8_prompt_audio,
                "S9": S9_prompt_audio,
                "S10": S10_prompt_audio,
            }
            for spk_key, audio in audio_inputs_simple.items():
                if audio is not None:
                    prompt_text = DEFAULT_PROMPTS.get(spk_key, f"{spk_key} default prompt")
                    speakers_data[spk_key] = {
                        "prompt_audio": audio,
                        "prompt_text": prompt_text,
                        "dialect_prompt": ""
                    }
        
        if not speakers_data:
            raise ValueError(
                "At least one speaker's audio is required!\n"
                "Please provide at least S1's prompt_audio and include corresponding dialogue content in dialogue_script.\n"
                "The system will automatically extract each speaker's first sentence from dialogue_script as prompt text.\n"
                "Example:\n[S1] Hello there, Xiaoxi.\n[S2] Hello, Nenglao!"
            )
        
        if not dialogue_script:
            raise ValueError("dialogue_script cannot be empty! Please enter a dialogue script, format: [S1] First sentence\n[S2] Second sentence")
        
        # 下方维持不变，始终用dialogue_script主流程
        text_list, spk_list = self._parse_dialogue_script(dialogue_script)
        
        used_spk_ids = set(spk_list)
        provided_spk_keys = set(speakers_data.keys())
        provided_spk_ids = {int(key[1:]) - 1 for key in provided_spk_keys}
        
        # Support up to MAX_SUPPORTED_SPEAKERS (defined at module level)
        invalid_spks = {spk_id for spk_id in used_spk_ids if spk_id < 0 or spk_id >= MAX_SUPPORTED_SPEAKERS}
        if invalid_spks:
            invalid_spk_labels = [f"S{spk_id+1}" for spk_id in sorted(invalid_spks)]
            raise ValueError(
                f"Unsupported speaker(s) used in dialogue script: {', '.join(invalid_spk_labels)}\n"
                f"Currently supports up to {MAX_SUPPORTED_SPEAKERS} speakers (S1 to S{MAX_SUPPORTED_SPEAKERS})."
            )
        
        missing_spks = used_spk_ids - provided_spk_ids
        if missing_spks:
            missing_spk_labels = [f"S{spk_id+1}" for spk_id in sorted(missing_spks)]
            raise ValueError(
                f"The following speaker(s) are used in the dialogue script but no corresponding audio is provided: {', '.join(missing_spk_labels)}\n"
                f"Provided speakers: {sorted(provided_spk_keys)}\n"
                f"Please ensure audio input is provided for all speakers used in the dialogue script."
            )
        
        # Get the maximum speaker ID used in the dialogue
        max_spk_id = max(used_spk_ids) if used_spk_ids else 0
        speaker_keys = [f"S{i+1}" for i in range(max_spk_id + 1)]
        
        prompt_wav_list = []
        prompt_text_list = []
        dialect_prompt_text_list = []
        
        for key in speaker_keys:
            if key in speakers_data:
                prompt_wav_list.append(speakers_data[key]["prompt_audio"])
                prompt_text_list.append(speakers_data[key]["prompt_text"])
                dialect_prompt_text_list.append(speakers_data[key]["dialect_prompt"])
        
        
        use_dialect_prompt = any(len(p) > 0 for p in dialect_prompt_text_list)
        
        prompt_text_ids_list = []
        dialect_prompt_text_ids_list = []
        spk_emb_list = []
        mel_list = []
        mel_len_list = []
        log_mel_list = []
        dialect_prefix_list = []
        
        dialect_prefix_list.append(tokenizer.encode(f"{TASK_PODCAST}"))
        
        for spk_idx, (prompt_text, prompt_audio) in enumerate(zip(prompt_text_list, prompt_wav_list)):
            if isinstance(prompt_audio, dict):
                audio_tensor = prompt_audio.get("waveform")
                sample_rate = prompt_audio.get("sample_rate")
                if audio_tensor is None or sample_rate is None:
                    raise ValueError(f"Invalid audio dictionary format, missing 'waveform' or 'sample_rate': {prompt_audio.keys()}")
            elif isinstance(prompt_audio, str):
                audio_tensor, sample_rate = torchaudio.load(prompt_audio)
            elif isinstance(prompt_audio, tuple) or isinstance(prompt_audio, list):
                if len(prompt_audio) >= 2:
                    audio_tensor, sample_rate = prompt_audio[0], prompt_audio[1]
                else:
                    raise ValueError(f"Invalid audio format, expected (tensor, sample_rate), but got: {type(prompt_audio)}")
            else:
                raise ValueError(f"Unsupported audio format: {type(prompt_audio)}")
            
            if not isinstance(audio_tensor, torch.Tensor):
                raise ValueError(f"Invalid audio tensor type, expected torch.Tensor, but got: {type(audio_tensor)}")
            
            if not isinstance(sample_rate, (int, float)):
                raise ValueError(f"Invalid sample rate type, expected int/float, but got: {type(sample_rate)}")
            sample_rate = int(sample_rate)
            
            # Normalize audio tensor to 1D: [samples]
            # Handle various input shapes: [channels, samples], [1, samples], [samples], [1, channels, samples], etc.
            original_shape = audio_tensor.shape
            while audio_tensor.dim() > 1:
                if audio_tensor.dim() == 2:
                    # For shape [channels, samples] or [1, samples]
                    if audio_tensor.shape[0] == 1:
                        # Mono audio with batch dimension: [1, samples] -> [samples]
                        audio_tensor = audio_tensor.squeeze(0)
                    else:
                        # Multi-channel audio: [channels, samples] -> take first channel [samples]
                        audio_tensor = audio_tensor[0]
                else:
                    # For 3D+ tensors, squeeze or take first element
                    audio_tensor = audio_tensor.squeeze()
                    # If still not 1D after squeeze, take first element along first dimension
                    if audio_tensor.dim() > 1:
                        audio_tensor = audio_tensor[0]
            
            if audio_tensor.dim() != 1:
                raise ValueError(f"Invalid audio tensor dimensions after processing: {audio_tensor.shape} (original: {original_shape})")
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_for_tokenizer = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
            else:
                audio_for_tokenizer = audio_tensor
            
            audio_for_tokenizer = audio_volume_normalize(audio_for_tokenizer)
            log_mel = s3tokenizer.log_mel_spectrogram(audio_for_tokenizer)
            
            spk_feat = kaldi.fbank(audio_for_tokenizer.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
            spk_emb = spk_model.run(
                None, {spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()
            
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
                audio_for_flow = resampler(audio_tensor.unsqueeze(0))
            else:
                audio_for_flow = audio_tensor.unsqueeze(0)
            
            audio_for_flow = audio_volume_normalize(audio_for_flow.squeeze(0)).unsqueeze(0)
            mel = mel_spectrogram(audio_for_flow).transpose(1, 2).squeeze(0)
            if mel.shape[0] % 2 != 0:
                mel = mel[:-1]
            mel_len = mel.shape[0]
            
            prompt_text = normalize_text(prompt_text)
            formatted_prompt = f"{SPK_DICT[spk_idx]}{TEXT_START}{prompt_text}{TEXT_END}{AUDIO_START}"
            if spk_idx == 0:
                formatted_prompt = f"{TASK_PODCAST}{formatted_prompt}"
            prompt_text_ids = tokenizer.encode(formatted_prompt)
            prompt_text_ids_list.append(prompt_text_ids)
            
            if use_dialect_prompt and len(dialect_prompt_text_list[spk_idx]) > 0:
                dialect_prompt_text = normalize_text(dialect_prompt_text_list[spk_idx])
                dialect_prompt_text = f"{SPK_DICT[spk_idx]}{TEXT_START}{dialect_prompt_text}{TEXT_END}{AUDIO_START}"
                dialect_prompt_text_ids = tokenizer.encode(dialect_prompt_text)
                dialect_prompt_text_ids_list.append(dialect_prompt_text_ids)
                if spk_idx == 0:
                    dialect_prefix_list.append(tokenizer.encode(f"{TASK_PODCAST}"))
                else:
                    dialect_prefix_list.append([])
            else:
                dialect_prompt_text_ids_list.append([])
                if spk_idx == 0:
                    dialect_prefix_list.append(tokenizer.encode(f"{TASK_PODCAST}"))
                else:
                    dialect_prefix_list.append([])
            
            log_mel_list.append(log_mel)
            spk_emb_list.append(spk_emb)
            mel_list.append(mel)
            mel_len_list.append(mel_len)
        
        text_ids_list = []
        spk_ids_for_model = []
        for text, spk_id in zip(text_list, spk_list):
            text = normalize_text(text)
            
            if spk_id < 0 or spk_id >= MAX_SUPPORTED_SPEAKERS:
                raise ValueError(f"Unsupported speaker index used in dialogue script: {spk_id} (only 0 to {MAX_SUPPORTED_SPEAKERS-1} are supported, corresponding to S1 to S{MAX_SUPPORTED_SPEAKERS})")
            
            formatted_text = f"{SPK_DICT[spk_id]}{TEXT_START}{text}{TEXT_END}{AUDIO_START}"
            text_ids = tokenizer.encode(formatted_text)
            text_ids_list.append(text_ids)
            spk_ids_for_model.append(spk_id)
        
        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(log_mel_list)
        spk_emb_for_flow = torch.tensor(spk_emb_list)
        prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(mel_list, batch_first=True, padding_value=0)
        prompt_mels_lens_for_flow = torch.tensor(mel_len_list)
        
        podcast_input = {
            "prompt_mels_for_llm": prompt_mels_for_llm,
            "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
            "prompt_text_tokens_for_llm": prompt_text_ids_list,
            "text_tokens_for_llm": text_ids_list,
            "prompt_mels_for_flow_ori": prompt_mels_for_flow,
            "prompt_mels_lens_for_flow": prompt_mels_lens_for_flow,
            "spk_emb_for_flow": spk_emb_for_flow,
            "spk_ids": spk_ids_for_model,
            "use_dialect_prompt": use_dialect_prompt,
            "diff_spk_pause_ms": diff_spk_pause_ms,
        }
        
        if use_dialect_prompt:
            podcast_input.update({
                "dialect_prompt_text_tokens_for_llm": dialect_prompt_text_ids_list,
                "dialect_prefix": dialect_prefix_list,
            })
        
        return (podcast_input,)
    
    def _parse_dialogue_script(self, dialogue_script: str) -> tuple[List[str], List[int]]:
        """
        Parse dialogue script supporting:
        - Multiple speakers (S1-S10)
        - Inline pause tags <|pause:MS|> where MS is milliseconds
        - Multi-line text per speaker
        """
        text_list = []
        spk_list = []
        
        # Pattern to match speaker tags like [S1] through [S10] with non-greedy content capture
        # Note: This pattern is explicitly coded for S1-S10 matching MAX_SUPPORTED_SPEAKERS
        # If MAX_SUPPORTED_SPEAKERS changes, this pattern must be updated accordingly
        pattern = r'\[S([1-9]|10)\](.*?)(?=\[S(?:[1-9]|10)\]|$)'
        matches = list(re.finditer(pattern, dialogue_script, re.DOTALL))
        
        pause_token_pattern = re.compile(r'<\|pause:(\d+)\|>')
        
        for match in matches:
            spk_num_str = match.group(1)
            content = match.group(2).strip()
            
            if not content:
                continue
            
            try:
                spk_num = int(spk_num_str)
            except Exception:
                continue
            
            spk_id = spk_num - 1
            
            if spk_id < 0 or spk_id >= MAX_SUPPORTED_SPEAKERS:
                raise ValueError(f"Unsupported speaker identifier: S{spk_num}, currently supports S1-S{MAX_SUPPORTED_SPEAKERS}")
            
            # Split content by pause tags and process each segment
            parts = re.split(r'(<\|pause:\d+\|>)', content)
            
            current_text_parts = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                pause_match = pause_token_pattern.fullmatch(part)
                if pause_match:
                    # Append pause tag directly
                    current_text_parts.append(part)
                else:
                    # Regular text
                    current_text_parts.append(part)
            
            if current_text_parts:
                final_text = ' '.join(current_text_parts)
                text_list.append(final_text)
                spk_list.append(spk_id)
        
        if not text_list:
            raise ValueError("Dialogue script format error, failed to parse any dialogue content. Format should be: [S1] text content\n[S2] text content")
        
        return text_list, spk_list
    
    def _parse_json_config(self, json_config: str) -> tuple[Dict[str, Any], str]:
        import json as json_lib
        try:
            config = json_lib.loads(json_config)
            speakers = config.get("speakers", {})
            
            dialogue_script = config.get("dialogue_script", "")
            
            if not dialogue_script:
                text_entries = config.get("text", [])
                if text_entries:
                    dialogue_lines = []
                    for entry in text_entries:
                        if isinstance(entry, list) and len(entry) >= 2:
                            spk_name, utt_text = entry[0], entry[1]
                            if spk_name.startswith("S") and spk_name[1:].isdigit():
                                dialogue_lines.append(f"[{spk_name}]{utt_text}")
                    dialogue_script = "\n".join(dialogue_lines)
            
            speakers_data = {}
            for spk_name, spk_data in speakers.items():
                speakers_data[spk_name] = {
                    "prompt_text": spk_data.get("prompt_text", ""),
                    "dialect_prompt": spk_data.get("dialect_prompt", ""),
                }
            
            return speakers_data, dialogue_script
        except json_lib.JSONDecodeError as e:
            raise ValueError(f"JSON config parsing failed: {e}")


class SoulXPodcastGenerate:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soulx_model": ("SOULX_MODEL",),
                "podcast_input": ("PODCAST_INPUT",),
                "seed": ("INT", {
                    "default": 1988,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.25,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "top_k": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),
                "min_tokens": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "max_tokens": ("INT", {
                    "default": 3000,
                    "min": 100,
                    "max": 5000,
                    "step": 100
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO",)
    RETURN_NAMES = ("combined_audio", "speaker_1_audio", "speaker_2_audio", "speaker_3_audio", "speaker_4_audio", "speaker_5_audio", "speaker_6_audio", "speaker_7_audio", "speaker_8_audio", "speaker_9_audio", "speaker_10_audio",)
    FUNCTION = "generate"
    CATEGORY = "SoulX-Podcast"
    
    def generate(
        self,
        soulx_model: Dict[str, Any],
        podcast_input: Dict[str, Any],
        seed: int = 1988,
        temperature: float = 0.6,
        repetition_penalty: float = 1.25,
        top_k: int = 100,
        top_p: float = 0.9,
        min_tokens: int = 8,
        max_tokens: int = 3000,
    ):
        set_all_random_seed(seed)
        
        model = soulx_model["model"]
        
        sampling_params = SamplingParams(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            use_ras=True,
            win_size=25,
            tau_r=0.2,
        )
        
        forward_params = {
            "prompt_mels_for_llm": podcast_input["prompt_mels_for_llm"],
            "prompt_mels_lens_for_llm": podcast_input["prompt_mels_lens_for_llm"],
            "prompt_text_tokens_for_llm": podcast_input["prompt_text_tokens_for_llm"],
            "text_tokens_for_llm": podcast_input["text_tokens_for_llm"],
            "prompt_mels_for_flow_ori": podcast_input["prompt_mels_for_flow_ori"],
            "spk_emb_for_flow": podcast_input["spk_emb_for_flow"],
            "sampling_params": sampling_params,
            "spk_ids": podcast_input["spk_ids"],
            "use_dialect_prompt": podcast_input["use_dialect_prompt"],
        }
        
        if podcast_input["use_dialect_prompt"]:
            forward_params.update({
                "dialect_prompt_text_tokens_for_llm": podcast_input["dialect_prompt_text_tokens_for_llm"],
                "dialect_prefix": podcast_input["dialect_prefix"],
            })
        
        results_dict = model.forward_longform(**forward_params)
        
        # Get the pause duration between different speakers
        diff_spk_pause_ms = podcast_input.get("diff_spk_pause_ms", 0)
        spk_ids = podcast_input["spk_ids"]
        sample_rate = 24000
        
        # Track audio segments for each speaker (0-indexed: speaker 0 = S1, speaker 1 = S2, etc.)
        speaker_audio_segments = {i: [] for i in range(MAX_SUPPORTED_SPEAKERS)}
        
        target_audio = None
        for i, wav in enumerate(results_dict["generated_wavs"]):
            # Store this segment for the corresponding speaker
            if i < len(spk_ids):
                speaker_id = spk_ids[i]
                if 0 <= speaker_id < MAX_SUPPORTED_SPEAKERS:
                    speaker_audio_segments[speaker_id].append(wav.clone())
            
            if target_audio is None:
                target_audio = wav
            else:
                # Insert pause between different speakers if configured
                # Ensure we have valid indices for both current and previous speaker
                if diff_spk_pause_ms > 0 and i > 0 and len(spk_ids) > i:
                    prev_spk = spk_ids[i - 1]
                    curr_spk = spk_ids[i]
                    if prev_spk != curr_spk:
                        # Calculate silence length in samples
                        silence_len = int((diff_spk_pause_ms / 1000.0) * sample_rate)
                        if silence_len > 0:
                            # Create silence tensor matching target_audio's dimension
                            if target_audio.dim() == 2:
                                silence = torch.zeros((1, silence_len), dtype=target_audio.dtype, device=target_audio.device)
                                target_audio = torch.cat([target_audio, silence], dim=1)
                            elif target_audio.dim() == 3:
                                silence = torch.zeros((target_audio.shape[0], target_audio.shape[1], silence_len), dtype=target_audio.dtype, device=target_audio.device)
                                target_audio = torch.cat([target_audio, silence], dim=2)
                
                # Concatenate the audio segments
                if target_audio.dim() == 3:
                    if wav.dim() == 3:
                        target_audio = torch.cat([target_audio, wav], dim=2)
                    else:
                        wav = wav.unsqueeze(0)
                        target_audio = torch.cat([target_audio, wav], dim=2)
                elif target_audio.dim() == 2:
                    if wav.dim() == 2:
                        target_audio = torch.cat([target_audio, wav], dim=1)
                    else:
                        wav = wav.squeeze(0) if wav.shape[0] == 1 else wav[0]
                        target_audio = torch.cat([target_audio, wav], dim=1)
        
        # Process combined audio
        if target_audio.dim() == 2:
            audio_tensor = target_audio.unsqueeze(0)
        elif target_audio.dim() == 3:
            audio_tensor = target_audio
        else:
            audio_tensor = target_audio.unsqueeze(0) if target_audio.dim() == 1 else target_audio
        
        sample_rate = 24000
        
        combined_audio_output = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }
        
        # Create separated speaker audio outputs
        speaker_outputs = []
        for speaker_id in range(MAX_SUPPORTED_SPEAKERS):
            if speaker_audio_segments[speaker_id]:
                # Concatenate all segments for this speaker
                speaker_audio = None
                for seg in speaker_audio_segments[speaker_id]:
                    if speaker_audio is None:
                        speaker_audio = seg
                    else:
                        # Concatenate segments
                        if speaker_audio.dim() == 3:
                            if seg.dim() == 3:
                                speaker_audio = torch.cat([speaker_audio, seg], dim=2)
                            else:
                                seg = seg.unsqueeze(0)
                                speaker_audio = torch.cat([speaker_audio, seg], dim=2)
                        elif speaker_audio.dim() == 2:
                            if seg.dim() == 2:
                                speaker_audio = torch.cat([speaker_audio, seg], dim=1)
                            else:
                                seg = seg.squeeze(0) if seg.shape[0] == 1 else seg[0]
                                speaker_audio = torch.cat([speaker_audio, seg], dim=1)
                
                # Format to match ComfyUI audio format
                if speaker_audio.dim() == 2:
                    speaker_audio_tensor = speaker_audio.unsqueeze(0)
                elif speaker_audio.dim() == 3:
                    speaker_audio_tensor = speaker_audio
                else:
                    speaker_audio_tensor = speaker_audio.unsqueeze(0) if speaker_audio.dim() == 1 else speaker_audio
                
                speaker_outputs.append({
                    "waveform": speaker_audio_tensor,
                    "sample_rate": sample_rate
                })
            else:
                # No audio for this speaker - return None
                speaker_outputs.append(None)
        
        return (combined_audio_output, *speaker_outputs)


NODE_CLASS_MAPPINGS = {
    "SoulXPodcastLoader": SoulXPodcastLoader,
    "SoulXPodcastInputParser": SoulXPodcastInputParser,
    "SoulXPodcastGenerate": SoulXPodcastGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoulXPodcastLoader": "SoulX Podcast Loader",
    "SoulXPodcastInputParser": "SoulX Podcast Input Parser",
    "SoulXPodcastGenerate": "SoulX Podcast Generate",
}


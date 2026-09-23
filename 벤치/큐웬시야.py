"""(나) 로컬 생성기 — Qwen-Image-Edit-2511 Q4_K_S + fal Multiple-Angles LoRA 를 ComfyUI API 로 (09-23).

    python 벤치/큐웬시야.py <입력 png> <출력 png> ["<sks> right side view eye-level shot medium shot"]

  ComfyUI 가 127.0.0.1:8188 에 떠 있어야 한다. 모델: C:/ComfyUI_models (gguf · 텍스트 인코더 · vae), D:/ComfyUI_models/loras.
  라이선스: 모델 · LoRA 모두 Apache-2.0. 한 번에 한 시야라 시야끼리 일관성은 보장 안 됨 — 그래서 벤치로 잰다.
"""
import json
import sys
import time
import urllib.request

import os
import shutil

서버 = "http://127.0.0.1:8188"
입력폴더 = "D:/ComfyUI/input"


def 그래프(이름, 프롬프트, 걸음=20, cfg=4.0, 씨=0):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "qwen-image-edit-2511-multiple-angles-lora.safetensors", "strength_model": 0.9}},
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.1}},
        "4": {"class_type": "CFGNorm", "inputs": {"model": ["3", 0], "strength": 1.0}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": 이름}},
        "8": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["7", 0], "upscale_method": "lanczos", "megapixels": 1.0}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "vae": ["6", 0], "image1": ["8", 0], "prompt": 프롬프트}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "vae": ["6", 0], "image1": ["8", 0], "prompt": ""}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["9", 0], "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["10", 0], "reference_latents_method": "index_timestep_zero"}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["6", 0]}},
        "14": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["11", 0], "negative": ["12", 0], "latent_image": ["13", 0],
                                                  "seed": 씨, "steps": 걸음, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "genkei_view"}},
    }


def _get(경로):
    return json.load(urllib.request.urlopen(서버 + 경로))


def 만들기(입력, 출력, 프롬프트):
    이름 = "genkei_" + os.path.basename(입력)
    from PIL import Image
    im = Image.open(입력).convert("RGBA")
    흰 = Image.new("RGBA", im.size, "white")
    Image.alpha_composite(흰, im).convert("RGB").save(os.path.join(입력폴더, 이름))   # 투명 배경 -> 흰 배경
    req = urllib.request.Request(서버 + "/prompt", json.dumps({"prompt": 그래프(이름, 프롬프트)}).encode(), {"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    t = time.time()
    while True:
        h = _get("/history/" + pid)
        if pid in h:
            break
        time.sleep(5)
    st = h[pid]["status"]
    if st.get("status_str") != "success":
        raise RuntimeError(json.dumps(st, ensure_ascii=False)[:2000])
    im = h[pid]["outputs"]["16"]["images"][0]
    shutil.copy(os.path.join("D:/ComfyUI/output", im["subfolder"], im["filename"]), 출력)
    print("%s -> %s (%.0f 초)" % (프롬프트, 출력, time.time() - t))


if __name__ == "__main__":
    만들기(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "<sks> right side view eye-level shot medium shot")

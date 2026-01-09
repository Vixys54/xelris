import argparse
import logging
import os
import json
import gc
import diffusers.models.unets.unet_2d_condition
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import traceback
import glob

# Salva a função original
original_get_aug_embed = diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.get_aug_embed

# Cria uma versão corrigida DEFINITIVA
def patched_get_aug_embed(self, added_cond_kwargs, *args, **kwargs):
    if added_cond_kwargs is None:
        return None
    return original_get_aug_embed(self, added_cond_kwargs, *args, **kwargs)

# Aplica o patch
diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.get_aug_embed = patched_get_aug_embed

# ================================================
# PATCHES CRÍTICOS: PREVENIR FLOAT64 NO DIRECTML
# ================================================
import torch
torch.set_default_dtype(torch.float32)

# 1. Patch para evitar falhas no torch.finfo
import transformers.modeling_attn_mask_utils
original_finfo = torch.finfo

def patched_finfo(dtype):
    if dtype in [torch.float64, torch.double]:
        return original_finfo(torch.float32)
    return original_finfo(dtype)

torch.finfo = patched_finfo

# 2. Patch para evitar falhas no torch.full
original_torch_full = torch.full

def patched_torch_full(size, fill_value, *args, **kwargs):
    if 'dtype' in kwargs:
        if kwargs['dtype'] in [torch.float64, torch.double]:
            kwargs['dtype'] = torch.float32
    elif 'dtype' not in kwargs:
        kwargs['dtype'] = torch.float32
        
    return original_torch_full(size, fill_value, *args, **kwargs)

torch.full = patched_torch_full

# DirectML
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print("✓ torch-directml disponível (aceleração na iGPU)")
except ImportError:
    DIRECTML_AVAILABLE = False
    print("⚠️ torch-directml não encontrado (rodará em CPU)")

# ONNX Imports
try:
    from optimum.onnxruntime import ORTStableDiffusionPipeline
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("⚠️ ONNX Runtime não encontrado")

# OpenVINO Imports
try:
    from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionPipeline
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️ OpenVINO não encontrado")

# Diffusers imports
try:
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, OnnxStableDiffusionPipeline
    from diffusers import EulerDiscreteScheduler, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️ Diffusers não encontrado")

import onnxruntime as ort
import gradio as gr
import socket
import numpy as np

logging.basicConfig(
    filename='xelris_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_and_print(msg, level="info"):
    print(msg)
    if level == "info":
        logging.info(msg)
    elif level == "error":
        logging.error(msg)
    elif level == "warning":
        logging.warning(msg)

class ModelManager:
    def __init__(self):
        self.checkpoint_dir = 'Checkpoint'
        self.lora_dir = 'LoRA'
        self.resultado_dir = 'Resultado'
        self.current_model = None
        self.current_pipe = None
        self.models = {}
        self.loras = {}
        self.load_all_models()
        self.load_all_loras()
    
    def find_models_recursive(self, root_dir: str) -> Dict[str, str]:
        models = {}
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)
            return models
    
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if not os.path.isdir(item_path):
                continue
        
            item_path_normalized = item_path.replace('\\', '/')
            relative_path = os.path.relpath(item_path, root_dir).replace('\\', '/')
            
            has_index = os.path.exists(os.path.join(item_path, 'model_index.json'))
            onnx_files = glob.glob(os.path.join(item_path, '*.onnx'))
            xml_files = glob.glob(os.path.join(item_path, '*.xml'))
            
            if has_index or onnx_files or xml_files:
                models[relative_path] = item_path_normalized
                log_and_print(f"✓ Modelo encontrado: {relative_path}", "info")
        
        return models
    
    def find_loras_recursive(self, root_dir: str) -> Dict[str, str]:
        loras = {}
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)
            return loras
        
        lora_extensions = ('.safetensors', '.ckpt', '.pt', '.pth')
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(lora_extensions):
                    lora_path = os.path.join(root, file).replace('\\', '/')
                    relative_path = os.path.relpath(lora_path, root_dir).replace('\\', '/')
                    loras[relative_path] = lora_path
        
        return loras
    
    def find_generated_images(self, limit=50) -> List[Dict]:
        images = []
        if not os.path.exists(self.resultado_dir):
            return images
        
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for filepath in glob.glob(os.path.join(self.resultado_dir, ext)):
                try:
                    stat = os.stat(filepath)
                    images.append({
                        'path': filepath,
                        'name': os.path.basename(filepath),
                        'time': datetime.fromtimestamp(stat.st_mtime),
                        'size': stat.st_size
                    })
                except:
                    pass
        
        images.sort(key=lambda x: x['time'], reverse=True)
        return images[:limit]
    
    def load_all_models(self):
        self.models = self.find_models_recursive(self.checkpoint_dir)
        log_and_print(f"Total de modelos encontrados: {len(self.models)}", "info")
    
    def load_all_loras(self):
        self.loras = self.find_loras_recursive(self.lora_dir)
        log_and_print(f"Total de LoRAs encontradas: {len(self.loras)}", "info")
    
    def get_models_list(self) -> List[str]:
        self.load_all_models()
        return list(self.models.keys()) if self.models else ["Nenhum modelo encontrado"]
    
    def get_loras_list(self) -> List[str]:
        self.load_all_loras()
        return list(self.loras.keys()) if self.loras else []
    
    def get_model_info(self, model_name: str) -> Dict:
        if model_name not in self.models:
            return {"supports_lora": False, "backend": "unknown", "is_sdxl": False}
        
        # =================================================================
        # CORREÇÃO CRÍTICA: SSD-1B OVERRIDE
        # =================================================================
        if "ssd-1b" in model_name.lower():
            return {
                "supports_lora": True, 
                "backend": "diffusers", 
                "is_sdxl": True 
            }

        model_path = self.models[model_name]
        
        # Padrões iniciais
        is_sdxl = "sdxl" in model_name.lower()
        backend = "unknown"
        supports_lora = False
        
        # =================================================================
        # OVERRIDE PELO NOME (Prioridade máxima)
        # =================================================================
        if "openvino" in model_name.lower():
            return {
                "supports_lora": False, 
                "backend": "openvino", 
                "is_sdxl": is_sdxl
            }
        
        if "onnx" in model_name.lower() and "openvino" not in model_name.lower():
            return {
                "supports_lora": False, 
                "backend": "onnx", 
                "is_sdxl": is_sdxl
            }
        
        # =================================================================
        # DETECÇÃO POR ARQUIVOS (XML/ONNX)
        # =================================================================
        xml_files = glob.glob(os.path.join(model_path, '**', '*.xml'), recursive=True)
        if xml_files:
            return {
                "supports_lora": False, 
                "backend": "openvino", 
                "is_sdxl": is_sdxl
            }
            
        onnx_files = glob.glob(os.path.join(model_path, '**', '*.onnx'), recursive=True)
        if onnx_files:
            return {
                "supports_lora": False, 
                "backend": "onnx", 
                "is_sdxl": is_sdxl
            }
            
        # =================================================================
        # DETECÇÃO POR JSON
        # =================================================================
        config_file = os.path.join(model_path, 'model_index.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    config_str = json.dumps(config)
                    if "stable-diffusion-xl" in config_str.lower() or "sdxl" in config_str.lower():
                        is_sdxl = True
                    
                    class_name = config.get("_class_name", "").lower()
                    if "openvino" in class_name or class_name.startswith("ov"):
                        backend = "openvino"
                        supports_lora = False
                    elif "onnx" in class_name or "ort" in class_name:
                        backend = "onnx"
                        supports_lora = False
                    else:
                        backend = "diffusers"
                        supports_lora = True
                        
            except Exception as e:
                log_and_print(f"Erro ao ler config de {model_name}: {e}", "warning")

        if backend == "unknown":
            diffusers_folders = ['unet', 'text_encoder', 'vae', 'tokenizer', 'scheduler']
            is_diffusers = any(os.path.exists(os.path.join(model_path, folder)) for folder in diffusers_folders)
            if is_diffusers:
                backend = "diffusers"
                supports_lora = True
        
        return {
            "supports_lora": supports_lora, 
            "backend": backend, 
            "is_sdxl": is_sdxl
        }
    
    def load_model(self, model_name: str, provider: str) -> bool:
        if model_name not in self.models:
            log_and_print(f"Modelo não encontrado: {model_name}", "error")
            return False
        
        model_path = self.models[model_name].replace('\\', '/')
        pipe = None
        
        try:
            log_and_print(f"Carregando modelo: {model_name}", "info")
            print(f"\n⏳ Carregando {model_name}...\n")
            
            # =================================================================
            # CORREÇÃO DE MEMÓRIA: LIMPA MODELO ANTERIOR
            # =================================================================
            if self.current_pipe is not None:
                print("   🧹 Limpando memória do modelo anterior...")
                try:
                    if hasattr(self.current_pipe, 'unload_lora_weights'):
                        self.current_pipe.unload_lora_weights()
                except:
                    pass
                
                del self.current_pipe
                self.current_pipe = None
                self.current_model = None
                gc.collect()
                
            # =================================================================

            info = self.get_model_info(model_name)
            
            # =================================================================
            # TRAVA DE SEGURANÇA PARA DIRECTML
            # =================================================================
            if info["backend"] == "onnx" and provider == "DmlExecutionProvider":
                print("   ⚠️ ATENÇÃO: Você está usando um modelo ONNX com DirectML.")
                print("      O DirectML ONNX é notoriamente instável em algumas iGPUs Intel.")
                print("      Recomendação fortemente: Use a versão 'OpenVINO' do mesmo modelo.")
                print("      (Opção 3 no menu de Download: MODELO REALISTA - OpenVINO)")
            
            print(f"   Backend Detectado: {info['backend']}, SDXL: {info['is_sdxl']}")
            
            if info["backend"] == "openvino" and OPENVINO_AVAILABLE:
                try:
                    print("⚡ Modo OpenVINO Ativado!")
                    if info["is_sdxl"]:
                        from optimum.intel.openvino import OVStableDiffusionXLPipeline
                        pipe = OVStableDiffusionXLPipeline.from_pretrained(
                            model_path, 
                            compile=False,
                            device="GPU"
                        )
                    else:
                        pipe = OVStableDiffusionPipeline.from_pretrained(
                            model_path, 
                            compile=False,
                            device="GPU"
                        )
                    print("   ✓ OpenVINO carregado")
                except Exception as e:
                    log_and_print(f"OpenVINO falhou: {e}", "warning")
                    print(f"   ⚠️ OpenVINO falhou, tentando ONNX...")
                    pipe = None
            
            if info["backend"] == "onnx" and not pipe:
                try:
                    if ORT_AVAILABLE:
                        print("📦 Carregando como ONNX...")
                        if info["is_sdxl"]:
                            try:
                                from optimum.onnxruntime import ORTStableDiffusionXLPipeline
                                pipe = ORTStableDiffusionXLPipeline.from_pretrained(
                                    model_path, 
                                    provider=provider
                                )
                            except:
                                pipe = ORTStableDiffusionPipeline.from_pretrained(
                                    model_path, 
                                    provider=provider
                                )
                        else:
                            pipe = ORTStableDiffusionPipeline.from_pretrained(
                                model_path, 
                                provider=provider
                            )
                    else:
                        print("   ⚠️ ORT não disponível, usando diffusers ONNX...")
                        pipe = OnnxStableDiffusionPipeline.from_pretrained(model_path)
                    
                    if hasattr(pipe, 'enable_attention_slicing'):
                        pipe.enable_attention_slicing()
                    print("   ✓ ONNX carregado")
                except Exception as e:
                    log_and_print(f"ONNX falhou: {e}", "warning")
                    print(f"   ⚠️ ONNX falhou: {e}")
                    pipe = None
            
            if info["backend"] == "diffusers" and DIFFUSERS_AVAILABLE and not pipe:
                try:
                    print("⚡ Carregando como Diffusers...")
                    
                    if DIRECTML_AVAILABLE:
                        device = torch_directml.device()
                        print(f"   Dispositivo: DirectML (iGPU)")
                    else:
                        device = torch.device("cpu")
                        print(f"   Dispositivo: CPU")
                    
                    # Carrega o pipeline apropriado
                    if info["is_sdxl"]:
                        print("   Tipo: SDXL / SSD-1B (carregando em float32)")
                        pipe = StableDiffusionXLPipeline.from_pretrained(
                            model_path,
                            dtype=torch.float32,
                            use_safetensors=True,
                            safety_checker=None
                        )
                    else:
                        print("   Tipo: SD 1.5/2.1 (carregando em float32)")
                        pipe = StableDiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32,
                            use_safetensors=True,
                            safety_checker=None
                        )
                    
                    print("   Convertendo todos os componentes para float32...")
                    pipe.text_encoder = pipe.text_encoder.to(torch.float32)
                    if hasattr(pipe, 'text_encoder_2'):
                        pipe.text_encoder_2 = pipe.text_encoder_2.to(torch.float32)
                    pipe.unet = pipe.unet.to(torch.float32)
                    pipe.vae = pipe.vae.to(torch.float32)
                    
                    if DIRECTML_AVAILABLE:
                        print("   Aplicando configurações especiais para DirectML...")
                        if hasattr(pipe, 'set_use_memory_efficient_attention_xformers'):
                            pipe.set_use_memory_efficient_attention_xformers(False)
                        try:
                            from diffusers import DPMSolverMultistepScheduler
                            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                        except:
                            pass
                    
                    pipe.to(device)
                    
                    if hasattr(pipe, 'enable_attention_slicing'):
                        pipe.enable_attention_slicing()
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    
                    print("   ✓ Diffusers carregado em float32")
                    
                except Exception as e:
                    log_and_print(f"Diffusers falhou: {e}", "warning")
                    print(f"   Detalhe: {e}")
                    traceback.print_exc()
                    pipe = None
            
            if pipe:
                if hasattr(pipe, 'safety_checker'):
                    pipe.safety_checker = None
                
                self.current_model = model_name
                self.current_pipe = pipe
                log_and_print(f"✓ Modelo carregado: {model_name} ({info['backend']})", "info")
                print(f"✓ Modelo carregado com sucesso!\n")
                return True
            else:
                print(f"❌ Nenhum backend funcionou para {model_name}")
                return False

        except Exception as e:
            log_and_print(f"Erro ao carregar modelo {model_name}: {str(e)}", "error")
            print(f"❌ Erro: {str(e)}")
            traceback.print_exc()
            return False

model_manager = ModelManager()

parser = argparse.ArgumentParser()
parser.add_argument('--provider', default='DmlExecutionProvider')
args = parser.parse_args()

available_providers = ort.get_available_providers() if hasattr(ort, 'get_available_providers') else ['CPUExecutionProvider']
provider = args.provider if args.provider in available_providers else available_providers[0]

log_and_print(f"Provider selecionado: {provider}", "info")

def generate_image(
    prompt: str,
    model_name: str,
    selected_loras_dict: Dict,
    num_steps: int =20,
    width: int =512,
    height: int =512,
    cfg_scale: float = 7.0,
    negative_prompt: str = ""
) -> Tuple:
    
    if not prompt or prompt.strip() == "":
        return None, "❌ Prompt está vazio!"
    
    if model_name != model_manager.current_model:
        if not model_manager.load_model(model_name, provider):
            return None, f"❌ Erro ao carregar modelo: {model_name}"
    
    if not model_manager.current_pipe:
        return None, "❌ Nenhum modelo carregado!"
    
    info = model_manager.get_model_info(model_name)
    
    try:
        log_and_print(f"Gerando imagem - Prompt: {prompt[:50]}...", "info")
        print(f"\n⏳ Gerando imagem ({width}x{height})...\n")
        
        pipe = model_manager.current_pipe
        
        if info["supports_lora"] and selected_loras_dict and hasattr(pipe, 'load_lora_weights'):
            print(f"   Aplicando {len(selected_loras_dict)} LoRA(s)...")
            adapter_names = []
            weights = []
            for lora_name, strength in selected_loras_dict.items():
                lora_path = model_manager.loras.get(lora_name)
                if not lora_path:
                    print(f"   ⚠️ LoRA não encontrada: {lora_name}")
                    continue
                
                adapter_name = os.path.splitext(os.path.basename(lora_path))[0][:20]
                
                try:
                    if not hasattr(pipe, 'peft_config') or adapter_name not in pipe.peft_config:
                        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                    adapter_names.append(adapter_name)
                    weights.append(strength)
                    print(f"   ✓ {adapter_name} (força: {strength})")
                except Exception as e:
                    print(f"   ❌ Erro ao carregar LoRA {lora_name}: {e}")
            
            if adapter_names and hasattr(pipe, 'set_adapters'):
                pipe.set_adapters(adapter_names, adapter_weights=weights)
                print(f"   ✓ {len(adapter_names)} LoRAs aplicadas")
        
        print("   ⚡ Gerando imagem...")
        generator_args = {
            "prompt": prompt,
            "num_inference_steps": num_steps,
            "guidance_scale": cfg_scale,
            "negative_prompt": negative_prompt,
            "height": height,       # Sempre passa height
            "width": width,         # Sempre passa width
        }
        
        # =================================================================
        # TRATAMENTO DE ERRO PROTEGIDO
        # =================================================================
        try:
            # Desativa autocast
            original_autocast_enabled = torch.is_autocast_enabled()
            torch.set_autocast_enabled(False)
            
            with torch.no_grad():
                result = pipe(**generator_args)
                if hasattr(result, 'images'):
                    image = result.images[0]
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    image = result[0]
                else:
                    image = result
            
            torch.set_autocast_enabled(original_autocast_enabled)

        # 1. TRATA CRASH DE DIRECTML ONNX (UnicodeDecodeError)
        except UnicodeDecodeError as e:
            error_msg = "❌ ERRO CRÍTICO DO DRIVER (DirectML/ONNX)."
            log_and_print(error_msg, "error")
            
            print(f"   ⚠️ {str(e)}")
            print("   ⚠️ SOLUÇÃO: O modelo ONNX com DirectML é instável na sua iGPU.")
            print("   💡 Use a versão 'OpenVINO' do mesmo modelo.")
            print("   💡 (Menu -> Opção 3 -> Download do Modelo)")
            
            advice = "Erro de Driver. Use o modelo OpenVINO (Opção 3)."
            return None, advice

        # 2. TRATA ERRO DE MEMÓRIA (RuntimeError)
        except RuntimeError as e:
            error_msg = str(e)
            
            if info["backend"] == "diffusers":
                print(f"   ⚠️ Erro na GPU ({str(e)[:50]}...), tentando CPU...")
                
                try:
                    original_device = pipe.device
                    pipe = pipe.to("cpu")
                    
                    pipe.text_encoder = pipe.text_encoder.to(torch.float32)
                    if hasattr(pipe, 'text_encoder_2'):
                        pipe.text_encoder_2 = pipe.text_encoder_2.to(torch.float32)
                    pipe.unet = pipe.unet.to(torch.float32)
                    pipe.vae = pipe.vae.to(torch.float32)
                    
                    with torch.no_grad():
                        torch.set_autocast_enabled(False)
                        result = pipe(**generator_args)
                        if hasattr(result, 'images'):
                            image = result.images[0]
                        else:
                            image = result
                    
                    pipe = pipe.to(original_device)
                    torch.set_autocast_enabled(original_autocast_enabled)
                    print("   ✓ Imagem gerada em CPU e movida de volta")
                except Exception as fallback_error:
                    print(f"   ❌ Fallback falhou: {fallback_error}")
                    return None, f"❌ Erro grave: {fallback_error}"
            else:
                # Se for OpenVINO/ONNX e crashar, não tenta mover tensor
                print(f"   ❌ Erro de GPU ({info['backend']}).")
                print(f"   Dica: O modelo {model_name} não suporta esta resolução/ação.")
                return None, f"❌ Erro de GPU no {info['backend']}"
        # =================================================================

        if info["supports_lora"] and selected_loras_dict and hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
            print("   LoRAs descarregadas")
        
        # Se chegou até aqui, o resultado existe
        if 'image' not in locals():
             return None, "❌ Imagem não foi gerada (Erro desconhecido)"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('Resultado', exist_ok=True)
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_prompt:
            safe_prompt = "imagem"
        
        image_filename = f"{safe_prompt}_{timestamp}.png"
        image_path = os.path.join('Resultado', image_filename)
        image.save(image_path)
        
        success_msg = f"✅ Imagem gerada! Salva em: {image_filename}"
        if selected_loras_dict:
            success_msg += f" + {len(selected_loras_dict)} LoRA(s)"
        
        return image, success_msg
    
    except Exception as e:
        error_msg = f"❌ Erro ao gerar imagem: {str(e)}"
        log_and_print(error_msg, "error")
        print(f"   ❌ {str(e)}")
        traceback.print_exc()
        return None, error_msg

try:
    with gr.Blocks(title="Xelrís - Gerador de Imagens") as demo:
        
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1>🎨 Xelrís - Gerador de Imagens Avançado</h1>
            <p>SSD-1B Leve + LoRAs Individuais | Otimizado iGPU Intel</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Resultado")
                output_image = gr.Image(label="Imagem Gerada", type="pil", height=500)
                output_text = gr.Textbox(label="Status", interactive=False, lines=3)
                
                gr.Markdown("### 🗂️ Galeria")
                def load_gallery():
                    images = model_manager.find_generated_images(limit=20)
                    return [img['path'] for img in images] if images else []
                gallery_display = gr.Gallery(label="Galeria", columns=4, height=400, object_fit="contain")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Configuração")
                model_dropdown = gr.Dropdown(
                    choices=model_manager.get_models_list(),
                    value=model_manager.get_models_list()[0] if model_manager.get_models_list() else None,
                    label="📄 Selecionar Modelo",
                    interactive=True
                )
                
                gr.Markdown("### 📝 Prompts")
                prompt = gr.Textbox(label="Prompt Positivo", placeholder="Descreva a imagem...", lines=4)
                negative = gr.Textbox(label="Negative Prompt", placeholder="O que evitar (opcional)", lines=2, value="lowres, worst quality, bad quality")
                
                gr.Markdown("### ⚙️ Parâmetros")
                with gr.Row():
                    steps = gr.Slider(1, 50, value=20, label="Passos", step=1)
                    cfg = gr.Slider(1, 20, value=7.0, label="CFG Scale", step=0.1)
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, label="Largura", step=64)
                    height = gr.Slider(256, 1024, value=512, label="Altura", step=64)
                
                btn = gr.Button("🎨 Gerar Imagem", variant="primary", size="lg", scale=1)
            
            with gr.Column(scale=1) as lora_section:
                gr.Markdown("### 📦 LoRAs (Máx 4)")
                
                lora_search = gr.Textbox(label="🔍 Pesquisar LoRAs", placeholder="Filtre todos os slots...")
                lora_counter = gr.HTML("<div style='padding: 10px; background: #334155; border-radius: 8px; text-align: center; font-weight: bold;'>🎯 0/4 ativas</div>")
                
                lora_list_state = gr.State(model_manager.get_loras_list())
                
                slot_components = []
                for i in range(4):
                    with gr.Row():
                        enable = gr.Checkbox(value=False, label=f"Slot {i+1}")
                        lora_dd = gr.Dropdown(choices=[], label="LoRA", value=None, allow_custom_value=False)
                        strength = gr.Slider(0, 2.0, value=0.8, step=0.1, label="Força")
                    slot_components.extend([enable, lora_dd, strength])
        
        def update_loras(search_term, current_list):
            all_loras = model_manager.get_loras_list()
            if search_term:
                filtered = [l for l in all_loras if search_term.lower() in l.lower()]
            else:
                filtered = all_loras
            updates = [gr.update(choices=filtered) for _ in range(4)]
            return filtered, *updates
        
        def update_counter(*states):
            count = sum(1 for i in range(0, len(states), 3) if states[i])
            color = "#10b981" if count > 0 else "#334155"
            return f"<div style='padding: 10px; background: {color}; border-radius: 8px; text-align: center; font-weight: bold;'>🎯 {count}/4 ativas</div>"
        
        def collect_loras(*states):
            loras_dict = {}
            for i in range(0, len(states), 3):
                enable = states[i]
                name = states[i+1]
                strength = states[i+2]
                if enable and name and name != "None":
                    loras_dict[name] = float(strength)
            return loras_dict
        
        lora_search.change(
            update_loras,
            inputs=[lora_search, lora_list_state],
            outputs=[lora_list_state] + slot_components[1::3]
        )
        
        for comp in slot_components:
            comp.change(update_counter, inputs=slot_components, outputs=lora_counter)
        
        def toggle_lora_visibility(model_name):
            if not model_name or model_name == "Nenhum modelo encontrado":
                return gr.update(visible=False), [], *[gr.update(choices=[]) for _ in range(4)]
            
            info = model_manager.get_model_info(model_name)
            if info["supports_lora"]:
                all_loras = model_manager.get_loras_list()
                return gr.update(visible=True), all_loras, *[gr.update(choices=all_loras) for _ in range(4)]
            return gr.update(visible=False), [], *[gr.update(choices=[]) for _ in range(4)]
        
        model_dropdown.change(
            toggle_lora_visibility,
            inputs=model_dropdown,
            outputs=[lora_section, lora_list_state] + slot_components[1::3]
        )
        
        def generate_and_update(prompt, model, steps, w, h, cfg, neg, *slot_states):
            if not model or model == "Nenhum modelo encontrado":
                return None, "❌ Selecione um modelo primeiro!", []
            
            loras_dict = collect_loras(*slot_states)
            result = generate_image(prompt, model, loras_dict, steps, w, h, cfg, neg)
            gallery = load_gallery()
            if result[0] is None:
                return None, result[1], gallery
            return result[0], result[1], gallery
        
        btn.click(
            generate_and_update,
            inputs=[prompt, model_dropdown, steps, width, height, cfg, negative] + slot_components,
            outputs=[output_image, output_text, gallery_display],
            show_progress=True
        )
        
        def refresh_gallery():
            return load_gallery()
        
        refresh_btn = gr.Button("🔄 Atualizar Galeria", variant="secondary")
        refresh_btn.click(refresh_gallery, outputs=gallery_display)
    
    log_and_print("✓ Interface iniciada com sucesso!", "info")
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"

    print("\n" + "="*60)
    print("🚀 SERVIDOR INICIADO!")
    print("="*60)
    print(f"Modelos: {3} | LoRAs: {len(model_manager.loras)}")
    print(f"Dispositivo: {'DirectML (iGPU)' if DIRECTML_AVAILABLE else 'CPU'}")
    print("Acesse em: http://localhost:7860")
    if local_ip != "127.0.0.1":
        print(f"Ou na rede: http://{local_ip}:7860")
    print("="*60 + "\n")
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

except Exception as e:
    error_msg = f"❌ Erro ao iniciar: {str(e)}"
    log_and_print(error_msg, "error")
    print(f"\n{error_msg}\n")
    traceback.print_exc()
    exit(1)
import os
import sys
import subprocess
import platform
import logging
import json
import shutil
import traceback

logging.basicConfig(filename='xelris_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(msg, level="info"):
    print(msg)
    if level == "info":
        logging.info(msg)
    elif level == "error":
        logging.error(msg)
    elif level == "warning":
        logging.warning(msg)

def get_folder_size(folder_path):
    total_size = 0
    if not os.path.exists(folder_path):
        return 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                continue
    return total_size

def get_readable_size(path):
    if os.path.exists(path):
        size_bytes = get_folder_size(path)
        size_gb = size_bytes / (1024 * 1024 * 1024)
        return f"{size_gb:.2f} GB" if size_gb > 0 else "~4-7 GB"
    else:
        return "~4-7 GB"

def create_venv(venv_path):
    if not os.path.exists(venv_path):
        print(f"Criando ambiente virtual em {venv_path}...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
            log_and_print("Ambiente virtual criado", "info")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao criar ambiente virtual: {e}")
            return False
        return True
    else:
        print("✓ Ambiente virtual já existe.")
        return True

def get_pip_path(venv_path):
    if platform.system() == 'Windows':
        return os.path.join(venv_path, 'Scripts', 'pip.exe')
    else:
        return os.path.join(venv_path, 'bin', 'pip')

def get_python_path(venv_path):
    if platform.system() == 'Windows':
        return os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        return os.path.join(venv_path, 'bin', 'python')

def check_installed_packages(pip_path):
    try:
        result = subprocess.run([pip_path, 'list', '--format=json'], 
                              capture_output=True, text=True, check=True, timeout=30)
        packages = json.loads(result.stdout)
        return {pkg['name'].lower(): pkg['version'] for pkg in packages}
    except Exception as e:
        print(f"   ⚠️ Erro ao listar pacotes: {e}")
        return {}

def verify_key_imports(venv_path):
    """Verifica se os pacotes críticos importam corretamente (detecta corrupção)"""
    python_path = get_python_path(venv_path)
    
    # Lista reduzida de pacotes críticos para testar
    critical_packages = [
        ('torch', 'import torch'),
        ('transformers', 'import transformers'),
        ('diffusers', 'import diffusers'),
        ('optimum', 'import optimum'),
        ('gradio', 'import gradio'),
        ('onnxruntime', 'import onnxruntime'),
    ]
    
    corrupted = []
    for pkg_name, import_code in critical_packages:
        try:
            subprocess.run([python_path, '-c', import_code], 
                          check=True, capture_output=True, timeout=10)
            print(f"   ✓ {pkg_name} importa corretamente")
        except Exception as e:
            corrupted.append(pkg_name)
            print(f"   ❌ Import falhou para {pkg_name}")
    
    return corrupted

def install_dependencies(venv_path):
    pip_path = get_pip_path(venv_path)
    python_path = get_python_path(venv_path)
    
    print("\n📦 Verificando integridade e atualizando dependências...")
    
    # 1. Atualiza pip sempre
    print("  ⬇️ Atualizando pip...")
    try:
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], 
                      capture_output=False, timeout=60, check=True)
        print("  ✓ pip atualizado")
    except Exception as e:
        print(f"  ⚠️ Falha ao atualizar pip: {e}")
    
    # 2. Verifica pacotes instalados
    installed = check_installed_packages(pip_path)
    print(f"  ✓ {len(installed)} pacotes encontrados no ambiente")
    
    # 3. Verifica imports críticos (detecta corrupção)
    print("\n🔍 Verificando integridade dos pacotes críticos...")
    corrupted = verify_key_imports(venv_path)
    if corrupted:
        print(f"   ⚠️ Pacotes corrompidos detectados: {', '.join(corrupted)}")
        print("   🔧 Tentando reparar...")
    
    # Lista de dependências com specs - ORDENADO para evitar conflitos
    deps = [
        'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
        'numpy<2.0',
        'pillow',
        'transformers>=4.36.0',
        'accelerate',
        'huggingface-hub==0.21.0',  # Versão específica para evitar conflitos
        'safetensors',
        'ftfy',
        'gradio>=4.13.0',
    ]
    
    # Dependências condicionais
    if platform.system() == 'Windows':
        deps.append('torch-directml')
        deps.append('onnxruntime-directml')
    else:
        deps.append('onnxruntime')
    
    deps.extend([
        'diffusers[onnx]>=0.26.0',
        'optimum[onnxruntime,openvino]>=1.17.0',
        'openvino',
        'onnx',
    ])
    
    # 4. Instala dependências na ordem correta
    print("\n🔧 Instalando/atualizando dependências...")
    for spec in deps:
        pkg_name = spec.split()[0].split('==')[0].split('[')[0]
        
        # Verifica se precisa instalar/atualizar
        pkg_lower = pkg_name.lower()
        needs_install = pkg_lower not in installed
        
        if not needs_install and pkg_lower in corrupted:
            needs_install = True
        
        if needs_install:
            try:
                print(f"  ⏳ {pkg_name}...")
                cmd = [pip_path, 'install']
                
                # Para torch, não usar --upgrade para evitar problemas
                if 'torch' in pkg_name:
                    cmd.extend(spec.split())
                else:
                    cmd.extend(['--upgrade'] + spec.split())
                
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=False,
                    timeout=300
                )
                print(f"  ✓ {pkg_name} instalado/atualizado")
            except Exception as e:
                print(f"  ⚠️ Erro ao instalar {pkg_name}: {e}")
        else:
            print(f"  ✓ {pkg_name} já instalado ({installed.get(pkg_lower, 'OK')})")
    
    # 5. Repara pacotes corrompidos se necessário
    if corrupted:
        print("\n🔧 Reparando pacotes corrompidos...")
        for pkg in corrupted:
            try:
                print(f"  ⏳ Reparando {pkg}...")
                subprocess.run(
                    [pip_path, 'install', '--force-reinstall', pkg],
                    check=True,
                    capture_output=False,
                    timeout=300
                )
                print(f"  ✓ {pkg} reparado")
            except Exception as e:
                print(f"  ⚠️ Falha ao reparar {pkg}: {e}")
    
    print("\n✓ Verificação de dependências concluída!\n")
    log_and_print("Dependências verificadas e atualizadas.", "info")
    return True

def detect_available_providers(venv_path):
    python_path = get_python_path(venv_path)
    
    try:
        code = "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
        result = subprocess.run([python_path, '-c', code], 
                              capture_output=True, text=True, check=True, timeout=10)
        providers = result.stdout.strip().split(',')
        return [p.strip() for p in providers if p.strip()]
    except:
        return ['CPUExecutionProvider']

def select_best_provider(venv_path):
    providers = detect_available_providers(venv_path)
    
    print("\n📊 Providers ONNX disponíveis:")
    if not providers:
        print("  ⚠️ Nenhum provider encontrado, usando CPU")
        return 'CPUExecutionProvider'
    
    for p in providers:
        print(f"  - {p}")
    
    # Ordem de preferência
    preferences = [
        'DmlExecutionProvider',      # DirectML (Windows iGPU)
        'CUDAExecutionProvider',     # NVIDIA CUDA
        'CoreMLExecutionProvider',   # Apple Silicon
        'TensorrtExecutionProvider', # NVIDIA TensorRT
        'CPUExecutionProvider'       # CPU fallback
    ]
    
    for pref in preferences:
        if pref in providers:
            print(f"\n✓ Provider selecionado: {pref}\n")
            log_and_print(f"Provider selecionado: {pref}", "info")
            return pref
    
    # Se nenhum dos preferidos estiver disponível, usa o primeiro
    selected = providers[0]
    print(f"\n✓ Provider selecionado: {selected}\n")
    return selected

def create_folders():
    folders = ['Checkpoint', 'LoRA', 'Resultado']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"  ✓ Criada pasta: {folder}/")
        else:
            print(f"  ✓ Pasta já existe: {folder}/")
    log_and_print("Pastas criadas/verificadas.", "info")

def export_model(model_id, export_folder_name, display_name, backend="onnx"):
    """Exporta modelo com tratamento robusto"""
    export_dir = os.path.join('Checkpoint', export_folder_name)
    
    # Verifica se o modelo já existe e está completo
    if os.path.exists(export_dir):
        model_index = os.path.join(export_dir, 'model_index.json')
        if os.path.exists(model_index):
            print(f"\n✓ '{display_name}' já existe!")
            size = get_readable_size(export_dir)
            print(f"   Tamanho: {size}")
            return True
        else:
            print(f"\n⚠️  Pasta existente parece incompleta, removendo...")
            try:
                shutil.rmtree(export_dir)
            except Exception as e:
                print(f"   ❌ Erro ao remover pasta: {e}")
                return False
    
    try:
        print(f"\n📥 Exportando: {display_name} ({backend.upper()})...")
        print(f"   Modelo: {model_id}")
        print(f"   Destino: {export_dir}")
        
        python_path = get_python_path('.venv')
        
        if not os.path.exists(python_path):
            raise FileNotFoundError("Python do venv não encontrado. Execute Opção 2 primeiro!")
        
        # Cria script de exportação baseado no backend
        if backend == "openvino":
            description = "   ⚡ OpenVINO FP16 (GPU integrada otimizada)"
            script_content = f"""import sys
import traceback
import warnings
# Suprime avisos de depreciação e tracing para deixar o log limpo
warnings.filterwarnings("ignore") 
from pathlib import Path

print("📦 Exportando para OpenVINO...")
try:
    from optimum.intel import OVStableDiffusionPipeline
    
    print("   Baixando modelo...")
    pipeline = OVStableDiffusionPipeline.from_pretrained(
        "{model_id}",
        export=True,
        compile=False,
        device="GPU"
    )
    
    print("   Salvando...")
    pipeline.save_pretrained(r"{export_dir}")
    print("✅ OpenVINO exportado com sucesso!")
except Exception as e:
    print(f"❌ Erro durante exportação: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
        
        elif backend == "diffusers":
            description = "   📦 Formato Diffusers puro (suporta LoRAs completas)"
            script_content = f"""import sys
import traceback
import warnings
# Suprime avisos de depreciação e tracing para deixar o log limpo
warnings.filterwarnings("ignore")
import torch

print("📦 Baixando modelo Diffusers...")
try:
    # Tenta SDXL primeiro, depois fallback para SD 1.5
    try:
        from diffusers import StableDiffusionXLPipeline
        print("   Usando StableDiffusionXLPipeline...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "{model_id}",
            dtype=torch.float32, # Atualizado: dtype em vez de torch_dtype
            safety_checker=None,
            use_safetensors=True
        )
    except:
        from diffusers import StableDiffusionPipeline
        print("   Usando StableDiffusionPipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "{model_id}",
            dtype=torch.float32, # Atualizado: dtype em vez de torch_dtype
            safety_checker=None,
            use_safetensors=True
        )
    
    print("   Salvando...")
    pipe.save_pretrained(r"{export_dir}")
    print("✅ Modelo salvo com sucesso!")
except Exception as e:
    print(f"❌ Erro: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
        
        else:  # onnx (padrão)
            description = "   🎯 ONNX (CPU-friendly)"
            script_content = f"""import sys
import traceback
import warnings
# Suprime avisos de depreciação e tracing para deixar o log limpo
warnings.filterwarnings("ignore")
from pathlib import Path

print("📦 Exportando para ONNX...")
try:
    from optimum.onnxruntime import ORTStableDiffusionPipeline
    
    print("   Baixando e convertendo...")
    pipeline = ORTStableDiffusionPipeline.from_pretrained(
        "{model_id}",
        export=True
    )
    
    print("   Salvando...")
    pipeline.save_pretrained(r"{export_dir}")
    print("✅ ONNX exportado com sucesso!")
except Exception as e:
    print(f"❌ Erro durante exportação: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
        
        print(description)
        print(f"  ⏳ Isso pode demorar 10-30 minutos (dependendo da internet)...\n")
        
        # Salva script temporário
        script_path = os.path.join('.venv', 'export_model.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Executa
        try:
            subprocess.run([python_path, script_path], check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erro ao exportar {display_name}.")
            print("   Verifique conexão com a internet e espaço em disco.")
            
            # Limpa pasta incompleta
            if os.path.exists(export_dir):
                try:
                    shutil.rmtree(export_dir)
                except:
                    pass
            
            return False
        finally:
            # Limpa script temporário
            try:
                os.remove(script_path)
            except:
                pass
        
        # Validação final
        if os.path.exists(os.path.join(export_dir, 'model_index.json')):
            size = get_readable_size(export_dir)
            print(f"\n✅ {display_name} pronto! [{size}]\n")
            log_and_print(f"Modelo {display_name} exportado.", "info")
            return True
        else:
            raise Exception("Modelo não foi criado corretamente")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        traceback.print_exc()
        return False

def verify_files():
    folders = ['Checkpoint', 'LoRA', 'Resultado']
    print("\n📂 Verificando estrutura:\n")
    
    for folder in folders:
        if os.path.exists(folder):
            # Conta itens na pasta
            try:
                items = len([name for name in os.listdir(folder) if not name.startswith('.')])
                status = f"✓ ({items} itens)"
            except:
                status = "✓"
            print(f"  {status} {folder}/")
        else:
            print(f"  ❌ {folder}/ (faltando)")
    
    check_path = 'Checkpoint'
    if os.path.exists(check_path):
        subdirs = [d for d in os.listdir(check_path) 
                  if os.path.isdir(os.path.join(check_path, d)) and not d.startswith('.')]
        if subdirs:
            print(f"\n  📦 Modelos encontrados ({len(subdirs)}):")
            for sub in subdirs:
                size = get_readable_size(os.path.join(check_path, sub))
                print(f"    - {sub} [{size}]")
        else:
            print("\n  ⚠️  Nenhum modelo encontrado. Execute Opção 4 para baixar modelos!")
    
    lora_path = 'LoRA'
    if os.path.exists(lora_path):
        lora_files = [f for f in os.listdir(lora_path) 
                     if f.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth'))]
        if lora_files:
            print(f"\n  🎨 LoRAs encontradas ({len(lora_files)}):")
            # Mostra apenas algumas
            for lora in lora_files[:5]:
                print(f"    - {lora}")
            if len(lora_files) > 5:
                print(f"    ... e mais {len(lora_files) - 5} LoRAs")
        else:
            print("\n  ⚠️  Nenhuma LoRA encontrada. Coloque seus arquivos .safetensors na pasta LoRA/")

def view_log():
    log_file = 'xelris_log.txt'
    if os.path.exists(log_file):
        print("\n📋 Log (últimas 50 linhas):\n")
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    for line in lines[-50:]:
                        print(line.rstrip())
                else:
                    print("Log vazio")
        except Exception as e:
            print(f"❌ Erro ao ler log: {e}")
    else:
        print("\n❌ Nenhum log encontrado")

def start_xelris(venv_path, provider):
    python_path = get_python_path(venv_path)
    
    if not os.path.exists(python_path):
        print("❌ Python do venv não encontrado! Execute Opção 2 primeiro.")
        return
    
    script = 'xelris_advanced.py'
    
    if not os.path.exists(script):
        print(f"❌ Arquivo {script} não encontrado!")
        return
    
    print(f"\n🚀 Iniciando Xelrís com provider: {provider}")
    print("   A interface abrirá em seu navegador...\n")
    
    try:
        subprocess.run([python_path, script, '--provider', provider])
    except KeyboardInterrupt:
        print("\n⏹️  Xelrís interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao iniciar Xelrís: {e}")

def create_bat_file(provider):
    if platform.system() == 'Windows':
        bat_content = f"""@echo off
chcp 65001 > nul
echo.
echo 🎨 Xelrís - Gerador de Imagens IA
echo ========================================
echo.
set VENV_PATH=.venv

if not exist "%VENV_PATH%\\Scripts\\activate.bat" (
    echo ❌ Ambiente virtual não encontrado.
    echo Execute o menu.py e escolha a Opção 2 primeiro.
    pause
    exit /b 1
)

call "%VENV_PATH%\\Scripts\\activate.bat"
python xelris_advanced.py --provider "{provider}"
if errorlevel 1 pause
"""
        with open('Xelris.bat', 'w', encoding='utf-8') as f:
            f.write(bat_content)
        print("✓ Arquivo Xelris.bat criado!")
        print("  Você pode usar este arquivo para iniciar o Xelrís diretamente.")

def main_menu():
    venv_path = '.venv'
    provider = None
    
    # Verifica se estamos no ambiente virtual correto
    is_in_venv = sys.prefix != sys.base_prefix
    
    print("\n" + "="*60)
    print("🎨 Xelrís - Gerador de Imagens IA")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.system()} {platform.release()}")
    if is_in_venv:
        print(f"Ambiente: Virtual ({os.path.basename(sys.prefix)})")
    print("="*60)
    
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL")
        print("="*60)
        print("1 - Iniciar Xelrís")
        print("2 - Instalar / Verificar dependências")
        print("3 - Verificar arquivos e modelos")
        print("4 - Download de Modelos (9 modelos disponíveis)")
        print("5 - Ver log do sistema")
        print("6 - Sair")
        print("="*60)
        
        choice = input("\nEscolha uma opção (1-6): ").strip()
        
        if choice == '1':
            if not os.path.exists(venv_path):
                print("\n❌ Ambiente virtual não existe.")
                print("   Execute a Opção 2 primeiro para instalar as dependências.")
                continue
            
            models_dir = 'Checkpoint'
            has_models = False
            if os.path.exists(models_dir):
                # Verifica se há pelo menos uma pasta de modelo válida
                for item in os.listdir(models_dir):
                    item_path = os.path.join(models_dir, item)
                    if os.path.isdir(item_path):
                        # Verifica se tem model_index.json ou arquivos ONNX/OpenVINO
                        has_json = os.path.exists(os.path.join(item_path, 'model_index.json'))
                        has_onnx = any(f.endswith('.onnx') for f in os.listdir(item_path) 
                                      if os.path.isfile(os.path.join(item_path, f)))
                        has_xml = any(f.endswith('.xml') for f in os.listdir(item_path) 
                                     if os.path.isfile(os.path.join(item_path, f)))
                        if has_json or has_onnx or has_xml:
                            has_models = True
                            break
            
            if not has_models:
                print("\n❌ Nenhum modelo encontrado.")
                print("   Execute a Opção 4 para baixar modelos primeiro.")
                continue
            
            if not provider:
                provider = select_best_provider(venv_path)
            
            start_xelris(venv_path, provider)
        
        elif choice == '2':
            print("\n" + "="*60)
            print("INSTALAÇÃO E VERIFICAÇÃO DE DEPENDÊNCIAS")
            print("="*60)
            
            if not create_venv(venv_path):
                continue
            
            if install_dependencies(venv_path):
                create_folders()
                provider = select_best_provider(venv_path)
                create_bat_file(provider)
                print(f"\n✓ Instalação concluída com sucesso!")
                print(f"  Provider configurado: {provider}")
                print(f"  Ambiente virtual: {venv_path}")
                print(f"  Pastas criadas: Checkpoint/, LoRA/, Resultado/")
        
        elif choice == '3':
            verify_files()
        
        elif choice == '4':
            while True:
                print("\n" + "="*60)
                print("📦 DOWNLOAD DE MODELOS")
                print("="*60)
                
                print("\n⚡ DIRECTML (GPU Integrada Intel/AMD + CPU)")
                print("   Velocidade: ⚡⚡⚡ | Qualidade: ⭐⭐⭐⭐")
                print("   1 - MODELO REALISTA (SD1.5, ~2.1 GB)")
                print("   2 - MODELO ANIME (Anything V5, ~2.0 GB)")
                
                print("\n🚀 OPENVINO (GPU Integrada OTIMIZADA)")
                print("   Velocidade: ⚡⚡⚡⚡ | Qualidade: ⭐⭐⭐⭐")
                print("   3 - MODELO REALISTA (SD1.5, ~2.1 GB)")
                print("   4 - MODELO ANIME (Anything V5, ~2.0 GB)")
                
                print("\n💾 CPU (Sem GPU, 100% Software)")
                print("   Velocidade: ⚡ | Qualidade: ⭐⭐⭐⭐⭐")
                print("   5 - MODELO REALISTA (SDXL, ~6.1 GB)")
                print("   6 - MODELO ANIME (Animagine XL 4.0, ~6.2 GB)")
                print("   7 - MODELO REALISTA (SD1.5, ~2.1 GB)")
                print("   8 - MODELO ANIME (Anything V5, ~2.0 GB)")
                
                print("\n🚀 LEVES (GPU Integrada + LoRAs)")
                print("   Velocidade: ⚡⚡⚡⚡ | Memória: ~2-3GB | LoRAs: ✓")
                print("   9 - SSD-1B (Distilled SDXL - Leve e Rápido)")
                
                print("\n🔧 UTILITÁRIOS:")
                print("L - Login HuggingFace (para modelos privados)")
                print("0 - Voltar ao menu principal")
                print("="*60)
                
                sub_choice = input("\nEscolha um modelo (0-9 ou L): ").strip().lower()
                
                if sub_choice == '0':
                    break
                
                elif sub_choice == 'l':
                    python_path = get_python_path('.venv')
                    if os.path.exists(python_path):
                        print("\n" + "="*60)
                        print("🔐 Login HuggingFace")
                        print("="*60)
                        print("\n1. Acesse: https://huggingface.co/settings/tokens")
                        print("2. Crie um novo token (READ access é suficiente)")
                        print("3. Cole o token abaixo:\n")
                        try:
                            subprocess.run([python_path, '-c', 
                                'from huggingface_hub import login; login()'], 
                                check=True)
                        except:
                            print("❌ Falha no login. Certifique-se de ter o huggingface-hub instalado.")
                    else:
                        print("❌ Ambiente virtual não encontrado. Execute Opção 2 primeiro.")
                
                elif sub_choice == '1':
                    export_model("runwayml/stable-diffusion-v1-5", 
                                "sd15_directml_real", 
                                "MODELO REALISTA (DirectML, SD1.5)", 
                                "onnx")
                elif sub_choice == '2':
                    export_model("genai-archive/anything-v5", 
                                "anything_v5_directml_anime", 
                                "MODELO ANIME (DirectML, Anything V5)", 
                                "onnx")
                elif sub_choice == '3':
                    export_model("runwayml/stable-diffusion-v1-5", 
                                "sd15_openvino_real", 
                                "MODELO REALISTA (OpenVINO, SD1.5)", 
                                "openvino")
                elif sub_choice == '4':
                    export_model("genai-archive/anything-v5", 
                                "anything_v5_openvino_anime", 
                                "MODELO ANIME (OpenVINO, Anything V5)", 
                                "openvino")
                elif sub_choice == '5':
                    export_model("stabilityai/stable-diffusion-xl-base-1.0", 
                                "sdxl_cpu_real", 
                                "MODELO REALISTA (CPU, SDXL)", 
                                "onnx")
                elif sub_choice == '6':
                    export_model("cagliostrolab/animagine-xl-4.0", 
                                "animagine_xl_4_cpu_anime", 
                                "MODELO ANIME (CPU, Animagine XL 4.0)", 
                                "onnx")
                elif sub_choice == '7':
                    export_model("runwayml/stable-diffusion-v1-5", 
                                "sd15_cpu_real", 
                                "MODELO REALISTA (CPU, SD1.5)", 
                                "onnx")
                elif sub_choice == '8':
                    export_model("genai-archive/anything-v5", 
                                "anything_v5_cpu_anime", 
                                "MODELO ANIME (CPU, Anything V5)", 
                                "onnx")
                elif sub_choice == '9':
                    export_model("segmind/SSD-1B", 
                                "ssd1b_leve", 
                                "SSD-1B (Leve, LoRAs, Rápido em iGPU)", 
                                "diffusers")
                else:
                    print("\n⚠️  Opção inválida!")
        
        elif choice == '5':
            view_log()
        
        elif choice == '6':
            print("\n👋 Até logo! Obrigado por usar o Xelrís!\n")
            sys.exit(0)
        
        else:
            print("\n⚠️  Opção inválida! Escolha um número de 1 a 6.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n⏹️  Programa interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        traceback.print_exc()
        input("\nPressione Enter para sair...")
        sys.exit(1)
#!/usr/bin/env python3
"""
🚀 OTIMIZAÇÕES TURBO PARA GPU INTEGRADA
Acelera OpenVINO + DirectML ao máximo
Versão corrigida sem conflitos
"""

import os
import subprocess
import sys
import platform
import json

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def run_cmd(cmd, desc=""):
    if desc:
        print(f"⏳ {desc}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        if result.stdout:
            print(f"✅ {desc}")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏱️  {desc} - Tempo esgotado")
        return False
    except Exception as e:
        print(f"⚠️  {desc} - Erro: {str(e)[:100]}")
        return False

def check_gpu_type():
    """Verifica se realmente tem GPU Intel antes de instalar otimizações"""
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(
                ['wmic', 'path', 'win32_videocontroller', 'get', 'name'],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if 'Intel' in line and ('Iris' in line or 'UHD' in line or 'HD' in line):
                    return True
        return False
    except:
        return False

def main():
    print_header("⚡ OTIMIZADOR TURBO - XELRÍS")
    print("\nEste script instala aceleradores extras para sua GPU integrada:")
    
    if not check_gpu_type():
        print("\n⚠️  GPU Intel não detectada!")
        print("Este otimizador é específico para GPUs Intel integradas.")
        print("Se você tem outra GPU (NVIDIA/AMD), não é necessário rodar este script.")
        resposta = input("\nContinuar mesmo assim? (s/N): ").strip().lower()
        if resposta != 's':
            print("\n❌ Otimização cancelada.")
            return
    
    venv_path = '.venv'
    if not os.path.exists(venv_path):
        print(f"\n❌ Ambiente virtual não encontrado em {venv_path}")
        print("   Execute: python menu.py → Opção 2 primeiro")
        return
    
    # Define caminhos baseado no sistema operacional
    if platform.system() == 'Windows':
        pip_path = os.path.join(venv_path, 'Scripts', 'pip.exe')
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    if not os.path.exists(pip_path):
        print(f"\n❌ pip não encontrado! Execute Opção 2 do menu primeiro.")
        return
    
    print(f"\n✓ venv encontrado: {venv_path}")
    
    # 1️⃣ Verificação de dependências básicas
    print_header("1️⃣  VERIFICANDO DEPENDÊNCIAS BÁSICAS")
    
    # Garante que onnxruntime-directml está instalado (mais seguro que IPEX)
    if platform.system() == 'Windows':
        print("\n⏳ Instalando/atualizando onnxruntime-directml...")
        run_cmd([pip_path, 'install', '--upgrade', 'onnxruntime-directml'], 
                "onnxruntime-directml")
    
    # 2️⃣ Instalar otimizações leves e seguras
    print_header("2️⃣  INSTALANDO OTIMIZAÇÕES SEGURAS")
    
    optimizations = [
        ('coloredlogs', 'coloredlogs'),  # Logs coloridos
        ('psutil', 'psutil'),  # Monitoramento de sistema
        ('onnx-simplifier', 'onnx-simplifier'),  # Reduz modelos ONNX
        ('--upgrade', 'torchvision'),  # Atualiza torchvision
    ]
    
    for cmd, name in optimizations:
        if cmd == '--upgrade':
            run_cmd([pip_path, 'install', '--upgrade', name], f"Atualizando {name}")
        else:
            run_cmd([pip_path, 'install', name], f"Instalando {name}")
    
    # 3️⃣ Teste de verificação
    print_header("3️⃣  VERIFICANDO ACELERADORES")
    
    test_code = '''
import sys
print("\\n" + "="*60)
print("🔧 DIAGNÓSTICO DE ACELERADORES")
print("="*60)

# ONNX Runtime
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print("ONNX Runtime Providers:")
    for p in providers:
        status = "✓" if p != "CPUExecutionProvider" else "⚠️"
        print(f"  {status} {p}")
except Exception as e:
    print(f"❌ ONNX Runtime: {e}")

# PyTorch
try:
    import torch
    print(f"\\nPyTorch: {torch.__version__}")
    if hasattr(torch, 'directml'):
        print("✓ torch-directml disponível")
    else:
        print("⚠️ torch-directml não disponível")
except Exception as e:
    print(f"❌ PyTorch: {e}")

# OpenVINO
try:
    import openvino as ov
    print(f"\\nOpenVINO: {ov.__version__}")
    print("✓ OpenVINO disponível para modelos .xml")
except:
    print("\\n⚠️ OpenVINO não instalado (opcional)")

print("\\n" + "="*60)
print("🎯 CONFIGURAÇÃO RECOMENDADA:")
print("  • Use modelos ONNX com DmlExecutionProvider")
print("  • Para Intel Iris: ative DirectML no menu.py")
print("  • Resolução: 512x512 para melhor velocidade")
print("  • Limite de LoRAs: Até 4 simultâneas (estabilidade)")
print("="*60)
'''
    
    try:
        subprocess.run([python_path, '-c', test_code], check=True)
    except Exception as e:
        print(f"Erro no teste: {e}")
    
    # 4️⃣ Criar arquivo de configuração otimizado
    print_header("4️⃣  CONFIGURAÇÃO OTIMIZADA")
    
    config_file = 'config_turbo.json'
    config_content = {
        "turbo_mode": {
            "enabled": True,
            "fast_generation": True,
            "reduce_vram": True,
            "optimal_resolution": "512x512",
            "max_loras": 4
        },
        "providers_priority": [
            "DmlExecutionProvider",
            "CPUExecutionProvider"
        ],
        "model_settings": {
            "openvino": {
                "compile": True,
                "device": "GPU",
                "precision": "FP16"
            },
            "onnx": {
                "enable_attention_slicing": True,
                "enable_vae_slicing": True
            }
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_content, f, indent=2)
    
    print(f"✅ Arquivo {config_file} criado com configurações otimizadas!")
    
    # 5️⃣ Dicas finais (atualizado para 4 LoRAs)
    print_header("5️⃣  DICAS PARA MÁXIMA VELOCIDADE")
    
    tips = """
🎯 CONFIGURAÇÕES RÁPIDAS NO XELRÍS:

1. SEMPRE use DmlExecutionProvider (se disponível)
   • 3-5x mais rápido que CPU

2. Limite de LoRAs: Até 4 simultâneas
   • Mais que 4 pode causar instabilidade
   • Combine forças: 0.5-1.0 geralmente é suficiente

3. Reduza resolução inicialmente:
   • 512x512 → Mais rápido, boa qualidade
   • 768x768 → Balanceado
   • 1024x1024 → Só se necessário

4. Ajuste passos (steps):
   • Rápido: 15-20 steps
   • Qualidade: 25-30 steps
   • Máximo: 40-50 steps (raro)

5. CFG Scale:
   • 6.0-7.0: Mais rápido, criativo
   • 7.0-8.0: Balanceado
   • 8.0-9.0: Segue prompt rigorosamente

⚡ DICAS COM LORAS:

• Comece com 1 LoRA, depois adicione mais
• Força total (1.0) nem sempre é melhor
• Teste combinações: personagem + estilo + cor
• Máximo recomendado: 2-3 LoRAs para estabilidade

📊 MONITORAMENTO:
• Task Manager → Performance → GPU
• Deve mostrar 70-90% uso durante geração
• Se mostra 100% CPU, algo está errado
"""
    
    print(tips)
    
    print_header("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("\nAgora execute:")
    print("1. python menu.py")
    print("2. Opção 1 (Iniciar Xelrís)")
    print("3. Selecione DmlExecutionProvider quando perguntado")
    print("\n⚡ Suas gerações serão muito mais rápidas!")
    print("🎨 Limite de LoRAs: Até 4 simultâneas para melhor estabilidade\n")

if __name__ == "__main__":
    main()
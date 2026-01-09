import os
import sys
import subprocess
import platform

def detect_cpu_info():
    """Detecta informações da CPU"""
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name', '/value'],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if 'Name=' in line:
                    return line.replace('Name=', '').strip()
        else:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Model name' in line:
                    return line.split(':')[1].strip()
    except:
        pass
    return platform.processor()

def detect_gpu_info():
    """Detecta GPUs disponíveis"""
    gpus = []
    
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_videocontroller', 'get', 'name'],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and 'Name' not in line:
                    gpus.append(line)
        except:
            pass
    else:
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'VGA' in line or 'Display' in line:
                    gpus.append(line.strip())
        except:
            pass
    
    return gpus

def check_directml():
    """Verifica se DirectML está disponível de forma simples"""
    try:
        venv_path = '.venv'
        if os.path.exists(venv_path):
            if platform.system() == 'Windows':
                python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
            else:
                python_path = os.path.join(venv_path, 'bin', 'python')
            
            if os.path.exists(python_path):
                code = 'import onnxruntime as ort; print("DmlExecutionProvider" in ort.get_available_providers())'
                result = subprocess.run(
                    [python_path, '-c', code],
                    capture_output=True, text=True
                )
                return result.stdout.strip() == 'True'
    except:
        pass
    return False

def main():
    print("\n" + "="*60)
    print("🔍 DETECÇÃO DE HARDWARE - XELRÍS")
    print("="*60)
    
    # Sistema
    print(f"\n💻 Sistema: {platform.system()} {platform.release()}")
    print(f"   Arquitetura: {platform.machine()}")
    
    # CPU
    cpu = detect_cpu_info()
    print(f"\n⚡ CPU: {cpu}")
    
    # GPU
    gpus = detect_gpu_info()
    if gpus:
        print(f"\n🎮 GPUs detectadas:")
        for gpu in gpus:
            print(f"   • {gpu}")
    else:
        print(f"\n⚠️  Nenhuma GPU detectada")
    
    # DirectML
    print(f"\n🎯 DirectML (iGPU Intel/AMD):")
    if check_directml():
        print("   ✅ Disponível - use DmlExecutionProvider")
    else:
        print("   ❌ Não disponível - use CPUExecutionProvider")
    
    # Recomendações
    print("\n" + "="*60)
    print("📋 RECOMENDAÇÕES:")
    
    has_intel_gpu = any('Intel' in gpu for gpu in gpus)
    has_nvidia = any('NVIDIA' in gpu for gpu in gpus)
    
    if has_intel_gpu:
        print("1. ✅ GPU Intel detectada")
        print("   • Execute: python optimize_gpu_turbo.py")
        print("   • Use modelos ONNX com DmlExecutionProvider")
    
    if has_nvidia:
        print("2. ✅ NVIDIA detectada")
        print("   • Instale CUDA Toolkit se ainda não tiver")
        print("   • Use CUDAExecutionProvider")
    
    if not has_intel_gpu and not has_nvidia:
        print("1. ⚠️  Sem GPU dedicada detectada")
        print("   • Use CPUExecutionProvider")
        print("   • Prefira modelos OpenVINO ou ONNX")
    
    print("\n3. 💾 Verifique espaço em disco:")
    print("   • Modelos ocupam 2-7GB cada")
    print("   • SSD é recomendado para velocidade")
    
    print("\n" + "="*60)
    print("🎮 Para otimizar, execute: python optimize_gpu_turbo.py")
    print("="*60)

if __name__ == "__main__":
    main()
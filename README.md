# Instalação 🇧🇷
# Xelrís V1
INSTAÇÂO AUTOMÁTICA PELO menu.py

RECOMENDAÇÕES (Obrigatorio): 

- Instale o python 3.11 ou superior
- Instale o Intel One API Tool
(Talvez precise instalar também o Visual Studio 2026)


\\\ ⚠️ Por favor, prefira usar a versão 2.0 que está melhorada e tem suporte a diversos modelos sem estourar a memória. ⚠️



# Xelrís 2.0 (Instalação)

Passo a passo
1. Instalar Python 3.11
No instalador, marcar "Add Python to PATH"
https://www.python.org/downloads/release/python-3118/

2. Instalar Git
Instalar com padrões
https://git-scm.com/download/win

4. Instalar Visual Studio Build Tools
Baixar o instalador
Selecionar "Desenvolvimento para desktop com C++"
Instalar (~6 GB)
https://visualstudio.microsoft.com/visual-cpp-build-tools/

6. Instalar Vulkan SDK
Baixar e instalar com padrões
Não precisa marcar opções extras
https://vulkan.lunarg.com/sdk/home

7. FFmpeg (opcional)	Para gerar vídeos MP4	
https://ffmpeg.org/download.html 
(ou winget install ffmpeg)




\\\ Como instalar o Xelrís 2.0 no Termux

- Passo 1, instale as dependências com os comandos a baixo, copie e cole no seu Termux.

```
pkg update && pkg upgrade
pkg install python clang cmake ninja ndk-sysroot vulkan-loader-android vulkan-headers libandroid-shmem-static
```





[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

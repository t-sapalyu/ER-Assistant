1. Download and install node.js
2. open terminal, go to the project folder - whatsapp-bot folder
3. run this 
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
4. npm install whatsapp-web.js qrcode-terminal axios
5. to launch: need 2 terminal open
    terminal 1: python .\app\rag_api.py
    terminal 2: node bot.js

# Cobre Quick Start Demo
# Generates: quickstart.gif

Require cobre

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"

Output quickstart.gif

Type "cobre init --template 1dtoy demo/"
Enter
Sleep 2s

Type "cobre run demo/"
Enter
Sleep 15s

Type "cobre report demo/output/"
Enter
Sleep 3s

Sleep 2s

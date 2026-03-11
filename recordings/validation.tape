# Cobre Validate Demo
# Generates: validation.gif

Require cobre

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"

Output validation.gif

Type "cobre init --template 1dtoy demo/"
Enter
Sleep 2s

Type "cobre validate demo/"
Enter
Sleep 5s

Sleep 2s

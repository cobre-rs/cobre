# Cobre Validate Demo
# Generates: validation.gif

Require cobre

Set Shell "bash"
Set FontSize 16
Set Width 100
Set Height 30
Set Theme "Catppuccin Mocha"

Output validation.gif

Type "cobre init --template 1dtoy demo/"
Enter
Sleep 2s

Type "cobre validate demo/"
Enter
Sleep 5s

Sleep 2s

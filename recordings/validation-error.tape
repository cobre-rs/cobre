# Cobre Validation Error Demo
# Generates: validation-error.gif
# Shows cobre validate stopping at a schema error before the reference layer runs.

Require cobre

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 14
Set Width 1200
Set Height 600
Set Theme { "background": "#0F1419", "foreground": "#C8C6C2", "cursor": "#B87333", "selection": "#1A2028", "black": "#0F1419", "red": "#DC4C4C", "green": "#4A8B6F", "yellow": "#F5A623", "blue": "#4A90B8", "magenta": "#D4956A", "cyan": "#4A8B6F", "white": "#C8C6C2", "brightBlack": "#2D3440", "brightRed": "#DC4C4C", "brightGreen": "#4A8B6F", "brightYellow": "#F5A623", "brightBlue": "#4A90B8", "brightMagenta": "#D4956A", "brightCyan": "#4A8B6F", "brightWhite": "#E8E6E3" }
Set WindowBar Colorful
Set BorderRadius 6
Set Padding 5
Set Margin 3
Set MarginFill "#2D3440"

Output validation-error.gif

Type "cobre init --template 1dtoy demo/"
Enter
Sleep 5s

# Schema error: remove the required 'reservoir' field from the first hydro
Type "jq '.hydros[0] |= del(.reservoir)' demo/system/hydros.json > tmp.json && mv tmp.json demo/system/hydros.json"
Enter
Sleep 5s

# Reference error: point the first thermal at a bus that does not exist
Type "jq '.thermals[0].bus_id = 99' demo/system/thermals.json > tmp.json && mv tmp.json demo/system/thermals.json"
Enter
Sleep 5s

Type "cobre validate demo/"
Enter
Sleep 5s

Sleep 2s

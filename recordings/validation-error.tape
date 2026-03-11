# Cobre Validation Error Demo
# Generates: validation-error.gif
# Shows cobre validate detecting a schema error and a semantic constraint violation.

Require cobre

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"

Output validation-error.gif

Type "cobre init --template 1dtoy demo/"
Enter
Sleep 2s

# Schema error: remove the required 'reservoir' field from the first hydro
Type "jq '.hydros[0] |= del(.reservoir)' demo/system/hydros.json > tmp.json && mv tmp.json demo/system/hydros.json"
Enter
Sleep 1s

# Semantic error: set max_storage_hm3 to a negative value on the second copy
# (also set a negative max_turbined_m3s to trigger a semantic constraint violation)
Type "jq '.hydros[0].generation.max_turbined_m3s = -5.0' demo/system/hydros.json > tmp.json && mv tmp.json demo/system/hydros.json"
Enter
Sleep 1s

Type "cobre validate demo/"
Enter
Sleep 5s

Sleep 2s

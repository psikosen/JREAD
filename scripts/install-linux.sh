#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
app_home="${XDG_DATA_HOME:-$HOME/.local/share}/JREAD"
app_dir="$app_home/app"
bin_dir="$app_home/bin"
host_name="com.jread.companion"
extension_id="$(tr -d '[:space:]' < "$project_dir/scripts/extension-id.txt")"
[[ "$extension_id" =~ ^[a-z0-9]+$ ]] || { echo "Invalid extension id" >&2; exit 1; }

cargo build --manifest-path "$project_dir/src-tauri/Cargo.toml" --release --bins
mkdir -p "$app_home" "$bin_dir" "${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ln -sfn "$project_dir" "$app_dir"
install -m 755 "$project_dir/src-tauri/target/release/jread-desktop" "$bin_dir/jread-desktop"
install -m 755 "$project_dir/src-tauri/target/release/jread-native-host" "$bin_dir/jread-native-host"

for browser_dir in \
  "$HOME/.config/google-chrome/NativeMessagingHosts" \
  "$HOME/.config/BraveSoftware/Brave-Browser/NativeMessagingHosts" \
  "$HOME/.config/chromium/NativeMessagingHosts"; do
  mkdir -p "$browser_dir"
  cat > "$browser_dir/$host_name.json" <<EOF
{
  "name": "$host_name",
  "description": "JREAD local companion launcher",
  "path": "$bin_dir/jread-native-host",
  "type": "stdio",
  "allowed_origins": ["chrome-extension://$extension_id/"]
}
EOF
done

cat > "${XDG_DATA_HOME:-$HOME/.local/share}/applications/jread.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=JREAD
Comment=Local Japanese subtitle companion
Exec=$bin_dir/jread-desktop
Terminal=false
Categories=AudioVideo;
EOF

echo "Installed JREAD for this user. Backend dependencies must be installed in $project_dir/.venv."

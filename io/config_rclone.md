# Rclone setup (Google Drive)

## 1. Install
brew install rclone

## 2. Configure
rclone config

- n → new remote
- name: gdrive
- type: drive
- login Google

## 3. Test
rclone lsd gdrive:

## 4. Usage
Upload:
rclone copy file.mp4 gdrive:MyDrive/input

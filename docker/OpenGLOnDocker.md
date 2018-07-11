

apt-get install -y x11-apps
### Install Xvfb
apt-get install xorg xvfb xfonts-100dpi xfonts-75dpi xfonts-scalable xfonts-cyrillic

### Create some kind of hidden screen
```
Xvfb :99 -screen 0 1280x1024x24  # No need for fancy options
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa/ DISPLAY=:99
```

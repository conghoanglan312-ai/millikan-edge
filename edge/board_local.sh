#!/bin/sh
# Set the board up to run the live session on its own screen: monitor on HDMI, mouse
# on USB, no laptop at any step.  Run once, as root:
#
#   adb push edge/live_board.py /root/millikan-edge/edge/
#   adb push edge/board_local.sh /tmp/ && adb shell sh /tmp/board_local.sh
#
# Needs on the board: /root/millikan-edge (cv/, edge/, results/) and the
# opencv-python 4.10.0.84 aarch64 wheel in /root (`pip download opencv-python==4.10.0.84
# --platform manylinux2014_aarch64 --only-binary=:all: --python-version 3.10 --no-deps`
# on the laptop; the board has no network).  The board logs in to the desktop user
# by itself (gdm3 automatic login), so power on -> desktop -> double-click a launcher.
# Closing the window ends the session, as q does.  Sessions land in ~/live.
set -e
# adb's shell runs with umask 0000: files it writes are writable by everyone, and
# GNOME refuses to start a launcher like that ("此桌面文件权限非法").
umask 022
APP=/opt/millikan-edge
GUI=/opt/cvgui
U=ztl
H=/home/$U
DESK=$(sudo -u $U xdg-user-dir DESKTOP)

mkdir -p $APP /opt/fixtures
cp -r /root/millikan-edge/cv /root/millikan-edge/edge /root/millikan-edge/results $APP/
[ -d $GUI/cv2 ] || pip3 install --no-deps --no-index --target $GUI \
    /root/opencv_python-4.10.0.84-*aarch64*.whl
[ -d /opt/fixtures/spray_labelled_0907 ] || \
    cp -r /root/test/fixtures/spray_labelled_0907 /opt/fixtures/

cat > $APP/run_local.sh <<EOF
#!/bin/sh
# The live session on this screen.  \$1: the source, cam:0 unless given.  One at a
# time: the session takes a few seconds to open its window, and a second double-click
# meanwhile started a second session beside it -- two windows, one session folder
# (2026-10-17).
mkdir -p $H/live
cd $APP
exec flock -n /tmp/millikan-live.lock env PYTHONPATH=$GUI python3 edge/live_board.py \\
    --local --source "\${1:-cam:0}" --out $H/live >> $H/live/last_session.log 2>&1
EOF
chmod +x $APP/run_local.sh
chown -R $U:$U $APP

launcher() {
    f="$DESK/$2"
    cat > "$f" <<EOF
[Desktop Entry]
Type=Application
Name=$1
Exec=$APP/run_local.sh $3
Terminal=false
Icon=camera-video
EOF
    chmod +x "$f"
    chown $U:$U "$f"
    # GNOME runs a desktop file only once it is marked trusted
    sudo -u $U DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
        gio set "$f" metadata::trusted true || true
}
mkdir -p "$DESK"
launcher "Millikan live" millikan-live.desktop cam:0
launcher "Millikan rehearsal" millikan-rehearsal.desktop \
    frames:/opt/fixtures/spray_labelled_0907
echo "launchers in $DESK; sessions -> $H/live"

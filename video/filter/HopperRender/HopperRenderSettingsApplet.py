#!/usr/bin/python3
import os
from time import sleep
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AppIndicator3', '0.1')
from gi.repository import Gtk, AppIndicator3, GLib

FIFO_PATH = "/tmp/hopperrender" # FIFO used to comunicate with the video filter
BUFFER_SIZE = 512               # Size of the buffer used to read from the FIFO
UPDATE_INTERVAL = 1/60          # Interval at which the update function will be called

# Callback functions for the radio items
def on_warped_frame12_activate(widget):
    if widget.get_active():
        print(1, flush=True)

def on_warped_frame21_activate(widget):
    if widget.get_active():
        print(2, flush=True)

def on_blended_frame_activate(widget):
    if widget.get_active():
        print(3, flush=True)

def on_hsvflow_activate(widget):
    if widget.get_active():
        print(4, flush=True)

def on_greyscaleflow_activate(widget):
    if widget.get_active():
        print(5, flush=True)

def on_sidebyside1_activate(widget):
    if widget.get_active():
        print(7, flush=True)

def on_sidebyside2_activate(widget):
    if widget.get_active():
        print(8, flush=True)

def on_tearingtest_activate(widget):
    if widget.get_active():
        print(9, flush=True)

def on_shader_bright_activate(widget):
    if widget.get_active():
        print(10, flush=True)

def on_shader_full_activate(widget):
    if widget.get_active():
        print(11, flush=True)

def on_shader_calm_activate(widget):
    if widget.get_active():
        print(12, flush=True)

def on_shader_off_activate(widget):
    if widget.get_active():
        print(13, flush=True)

def on_activation_toggle(widget):
    print(0, flush=True)

def quit_app(_):
    Gtk.main_quit()

# Main class for the AppIndicator
class HopperRenderSettings:
    def __init__(self):
        self.fd = os.open(FIFO_PATH, os.O_RDONLY | os.O_NONBLOCK)  # Open the FIFO for communication with the HopperRender filter

        # Create the AppIndicator
        script_dir = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(script_dir, 'hopperrendericon.png')
        indicator = AppIndicator3.Indicator.new(
            "HopperRenderSettings",
            icon_path,
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)

        # Main menu of the indicator
        main_menu = Gtk.Menu()

        # Status label used to display current information
        self.status_label = Gtk.Label(label="Interpolation Inactive")
        status_label_item = Gtk.MenuItem()
        status_label_item.add(self.status_label)
        status_label_item.show_all()
        main_menu.append(status_label_item)

        # Frame output menu
        frame_output_item = Gtk.MenuItem(label="Frame Output")
        main_menu.append(frame_output_item)
        frame_output_menu = Gtk.Menu()
        frame_output_item.set_submenu(frame_output_menu)
        
        # Shader menu
        shader_item = Gtk.MenuItem(label="Shader")
        main_menu.append(shader_item)
        shader_menu = Gtk.Menu()
        shader_item.set_submenu(shader_menu)

        # Frame output radio items
        frame_output_group = None

        warped_frame12_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "Warped Frame 1 -> 2")
        warped_frame12_item.connect("activate", on_warped_frame12_activate)
        frame_output_menu.append(warped_frame12_item)
        frame_output_group = warped_frame12_item.get_group()

        warped_frame21_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "Warped Frame 2 -> 1")
        warped_frame21_item.connect("activate", on_warped_frame21_activate)
        frame_output_menu.append(warped_frame21_item)
        frame_output_group = warped_frame21_item.get_group()

        blended_frame_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "Blended Frame")
        blended_frame_item.set_active(True)
        blended_frame_item.connect("activate", on_blended_frame_activate)
        frame_output_menu.append(blended_frame_item)
        frame_output_group = blended_frame_item.get_group()

        hsvflow_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "HSVFlow")
        hsvflow_item.connect("activate", on_hsvflow_activate)
        frame_output_menu.append(hsvflow_item)
        frame_output_group = hsvflow_item.get_group()

        greyscaleflow_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "GreyFlow")
        greyscaleflow_item.connect("activate", on_greyscaleflow_activate)
        frame_output_menu.append(greyscaleflow_item)
        frame_output_group = greyscaleflow_item.get_group()

        sidebyside1_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "SideBySide1")
        sidebyside1_item.connect("activate", on_sidebyside1_activate)
        frame_output_menu.append(sidebyside1_item)
        frame_output_group = sidebyside1_item.get_group()

        sidebyside2_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "SideBySide2")
        sidebyside2_item.connect("activate", on_sidebyside2_activate)
        frame_output_menu.append(sidebyside2_item)
        frame_output_group = sidebyside2_item.get_group()

        tearingtest_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "TearingTest")
        tearingtest_item.connect("activate", on_tearingtest_activate)
        frame_output_menu.append(tearingtest_item)
        frame_output_group = tearingtest_item.get_group()

        # Shader radio items
        shader_group = None

        shader_bright_item = Gtk.RadioMenuItem.new_with_label(shader_group, "0 - 220")
        shader_bright_item.connect("activate", on_shader_bright_activate)
        shader_menu.append(shader_bright_item)
        shader_group = shader_bright_item.get_group()

        shader_full_item = Gtk.RadioMenuItem.new_with_label(shader_group, "16 - 235")
        shader_full_item.connect("activate", on_shader_full_activate)
        shader_menu.append(shader_full_item)

        shader_calm_item = Gtk.RadioMenuItem.new_with_label(shader_group, "10 - 235")
        shader_calm_item.connect("activate", on_shader_calm_activate)
        shader_menu.append(shader_calm_item)

        shader_off_item = Gtk.RadioMenuItem.new_with_label(shader_group, "Off")
        shader_off_item.set_active(True)
        shader_off_item.connect("activate", on_shader_off_activate)
        shader_menu.append(shader_off_item)

        # Activation toggle
        activation_toggle = Gtk.CheckMenuItem(label="Activate")
        activation_toggle.set_active(True)
        activation_toggle.connect("toggled", on_activation_toggle)
        main_menu.append(activation_toggle)
        
        # Quit menu item
        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", quit_app)
        main_menu.append(quit_item)

        main_menu.show_all()

        indicator.set_menu(main_menu)
        GLib.idle_add(self.update)  # Add the update function that will listen on the FIFO for changes
        Gtk.main()

    # Update function that listens on the FIFO for changes
    def update(self):
        sleep(UPDATE_INTERVAL)  # Reduce the CPU usage by only refreshing every intermediate frame (assuming we are running on a 60hz display)
        try:
            data = os.read(self.fd, BUFFER_SIZE)
            if data:
                text = data.decode('utf-8')

                # Retreive the available calc resolutions and display them in the submenu
                if text.startswith("RES0"):
                    self.scalar_0_item.set_label(text[5:])
                elif text.startswith("RES1"):
                    self.scalar_1_item.set_label(text[5:])
                elif text.startswith("RES2"):
                    self.scalar_2_item.set_label(text[5:])
                elif text.startswith("RES3"):
                    self.scalar_3_item.set_label(text[5:])
                elif text.startswith("RES4"):
                    self.scalar_4_item.set_label(text[5:])
                elif text.startswith("RES5"):
                    self.scalar_5_item.set_label(text[5:])
                # Display the current status in the status label
                else:
                    self.status_label.set_text(text)
                
        except BlockingIOError:
            pass
        return True

if __name__ == "__main__":
    # Create the FIFO if it doesn't exist
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)

    # Initialize and start the AppIndicator
    applet = HopperRenderSettings()
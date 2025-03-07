#!/usr/bin/python3
import os
from time import sleep
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AppIndicator3', '0.1')
from gi.repository import Gtk, AppIndicator3, GLib

FIFO_PATH = "/tmp/hopperrender" # FIFO used to communicate with the video filter
BUFFER_SIZE = 512               # Size of the buffer used to read from the FIFO
UPDATE_INTERVAL = 1/60          # Interval at which the update function will be called

# Main class for the AppIndicator
class HopperRenderSettings:
    def __init__(self):
        self.black_level = 0
        self.white_level = 255
        self.delta_scalar = 8
        self.neighbor_scalar = 6

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

        # Level submenu
        level_slider_item = Gtk.MenuItem(label="Levels")
        level_slider_item.connect("activate", self.open_slider_window)
        main_menu.append(level_slider_item)

        # Shader menu
        shader_item = Gtk.MenuItem(label="Shader")
        main_menu.append(shader_item)
        shader_menu = Gtk.Menu()
        shader_item.set_submenu(shader_menu)

        # Frame output radio items
        frame_output_group = None

        warped_frame12_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "Warped Frame 1 -> 2")
        warped_frame12_item.connect("activate", self.on_warped_frame12_activate)
        frame_output_menu.append(warped_frame12_item)
        frame_output_group = warped_frame12_item.get_group()

        warped_frame21_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "Warped Frame 2 -> 1")
        warped_frame21_item.connect("activate", self.on_warped_frame21_activate)
        frame_output_menu.append(warped_frame21_item)
        frame_output_group = warped_frame21_item.get_group()

        blended_frame_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "Blended Frame")
        blended_frame_item.set_active(True)
        blended_frame_item.connect("activate", self.on_blended_frame_activate)
        frame_output_menu.append(blended_frame_item)
        frame_output_group = blended_frame_item.get_group()

        hsvflow_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "HSVFlow")
        hsvflow_item.connect("activate", self.on_hsvflow_activate)
        frame_output_menu.append(hsvflow_item)
        frame_output_group = hsvflow_item.get_group()

        greyscaleflow_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "GreyFlow")
        greyscaleflow_item.connect("activate", self.on_greyscaleflow_activate)
        frame_output_menu.append(greyscaleflow_item)
        frame_output_group = greyscaleflow_item.get_group()

        sidebyside1_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "SideBySide1")
        sidebyside1_item.connect("activate", self.on_sidebyside1_activate)
        frame_output_menu.append(sidebyside1_item)
        frame_output_group = sidebyside1_item.get_group()

        sidebyside2_item = Gtk.RadioMenuItem.new_with_label(frame_output_group, "SideBySide2")
        sidebyside2_item.connect("activate", self.on_sidebyside2_activate)
        frame_output_menu.append(sidebyside2_item)
        frame_output_group = sidebyside2_item.get_group()

        # Shader radio items
        shader_group = None

        no_shader_item = Gtk.RadioMenuItem.new_with_label(shader_group, "Off")
        no_shader_item.connect("activate", self.on_no_shader_activate)
        shader_menu.append(no_shader_item)
        shader_group = no_shader_item.get_group()

        shader1_item = Gtk.RadioMenuItem.new_with_label(shader_group, "10 - 219")
        shader1_item.connect("activate", self.on_shader1_activate)
        shader_menu.append(shader1_item)
        shader_group = shader1_item.get_group()

        shader2_item = Gtk.RadioMenuItem.new_with_label(shader_group, "16 - 219")
        shader2_item.connect("activate", self.on_shader2_activate)
        shader_menu.append(shader2_item)
        shader_group = shader2_item.get_group()

        # Activation toggle
        activation_toggle = Gtk.CheckMenuItem(label="Activate")
        activation_toggle.set_active(True)
        activation_toggle.connect("toggled", self.on_activation_toggle)
        main_menu.append(activation_toggle)
        
        # Quit menu item
        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", self.quit_app)
        main_menu.append(quit_item)

        main_menu.show_all()

        indicator.set_menu(main_menu)
        GLib.idle_add(self.update)  # Add the update function that will listen on the FIFO for changes
        Gtk.main()

    # Slider callback functions
    def on_black_level_slider_change(self, slider):
        self.black_level = int(slider.get_value())
        print(self.black_level + 100, flush=True)

    def on_white_level_slider_change(self, slider):
        self.white_level = int(slider.get_value())
        print(self.white_level + 400, flush=True)

    def delta_slider_change(self, slider):
        self.delta_scalar = int(slider.get_value())
        print(self.delta_scalar + 700, flush=True)

    def neighbor_slider_change(self, slider):
        self.neighbor_scalar = int(slider.get_value())
        print(self.neighbor_scalar + 800, flush=True)

    # Callback functions for the radio items
    def on_activation_toggle(self, widget):
        if widget.get_active():
            print(1, flush=True)
        else:
            print(0, flush=True)

    def on_warped_frame12_activate(self, widget):
        if widget.get_active():
            print(2, flush=True)

    def on_warped_frame21_activate(self, widget):
        if widget.get_active():
            print(3, flush=True)

    def on_blended_frame_activate(self, widget):
        if widget.get_active():
            print(4, flush=True)

    def on_hsvflow_activate(self, widget):
        if widget.get_active():
            print(5, flush=True)

    def on_greyscaleflow_activate(self, widget):
        if widget.get_active():
            print(6, flush=True)

    def on_sidebyside1_activate(self, widget):
        if widget.get_active():
            print(7, flush=True)

    def on_sidebyside2_activate(self, widget):
        if widget.get_active():
            print(8, flush=True)

    def on_no_shader_activate(self, widget):
        if widget.get_active():
            print(9, flush=True)

    def on_shader1_activate(self, widget):
        if widget.get_active():
            print(10, flush=True)
            self.black_level = 10
            self.white_level = 219

    def on_shader2_activate(self, widget):
        if widget.get_active():
            print(11, flush=True)
            self.black_level = 16
            self.white_level = 219

    def quit_app(_):
        Gtk.main_quit()

    # Function to open the level slider window
    def open_slider_window(self, source):
        # Create a new window
        self.window = Gtk.Window(title="Adjust Levels")
        self.window.set_default_size(300, 200)

        # Create a box to contain the slider
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.window.add(vbox)

        # Create the white level slider
        white_level_label = Gtk.Label(label="White Level")
        self.white_level_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.white_level_slider.set_value(self.white_level)
        self.white_level_slider.connect("value-changed", self.on_white_level_slider_change)

        # Create the black level slider
        black_level_label = Gtk.Label(label="Black Level")
        self.black_level_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.black_level_slider.set_value(self.black_level)
        self.black_level_slider.connect("value-changed", self.on_black_level_slider_change)

        # Create the delta scalar slider
        delta_scalar_label = Gtk.Label(label="Delta Scalar")
        self.delta_scalar_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 31, 1)
        self.delta_scalar_slider.set_value(self.delta_scalar)
        self.delta_scalar_slider.connect("value-changed", self.delta_slider_change)

        # Create the neighbor scalar slider
        neighbor_scalar_label = Gtk.Label(label="Neighbor Scalar")
        self.neighbor_scalar_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 31, 1)
        self.neighbor_scalar_slider.set_value(self.neighbor_scalar)
        self.neighbor_scalar_slider.connect("value-changed", self.neighbor_slider_change)

        # Add the slider to the box
        vbox.pack_start(white_level_label, True, True, 0)
        vbox.pack_start(self.white_level_slider, True, True, 0)
        vbox.pack_start(black_level_label, True, True, 0)
        vbox.pack_start(self.black_level_slider, True, True, 0)
        vbox.pack_start(delta_scalar_label, True, True, 0)
        vbox.pack_start(self.delta_scalar_slider, True, True, 0)
        vbox.pack_start(neighbor_scalar_label, True, True, 0)
        vbox.pack_start(self.neighbor_scalar_slider, True, True, 0)

        # Show all components
        self.window.show_all()

    # Update function that listens on the FIFO for changes
    def update(self):
        sleep(UPDATE_INTERVAL)  # Reduce the CPU usage by only refreshing every intermediate frame (assuming we are running on a 60hz display)
        try:
            data = os.read(self.fd, BUFFER_SIZE)
            if data:
                text = data.decode('utf-8')
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
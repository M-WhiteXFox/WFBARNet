import dearpygui.dearpygui as dpg


def apply_global_theme() -> None:
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (19, 24, 32))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (28, 36, 47))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (28, 36, 47))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (63, 74, 88))

            dpg.add_theme_color(dpg.mvThemeCol_Text, (232, 237, 242))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (148, 163, 184))

            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (15, 23, 42))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (36, 52, 71))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (51, 65, 85))

            dpg.add_theme_color(dpg.mvThemeCol_Button, (54, 79, 107))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (72, 102, 136))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (99, 121, 151))

            dpg.add_theme_color(dpg.mvThemeCol_Header, (33, 44, 59))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (50, 65, 85))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (71, 85, 105))

            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (34, 197, 94))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (59, 130, 246))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (96, 165, 250))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (16, 185, 129))

            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7)

    dpg.bind_theme(global_theme)

    with dpg.theme(tag="theme_button_primary"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (59, 130, 246))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (96, 165, 250))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (37, 99, 235))

    with dpg.theme(tag="theme_button_start"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (16, 185, 129))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (52, 211, 153))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (5, 150, 105))

    with dpg.theme(tag="theme_button_stop"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (239, 68, 68))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (248, 113, 113))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (220, 38, 38))

from __future__ import annotations

import abc

# Key actions follow the GLFW convention with the addition of DOWN and UP for edge detection.
RELEASE = 0  # Key is not being pressed
PRESS = 1  # Key is being pressed
REPEAT = 2  # Key is being held down (after initial press)
DOWN = 3  # Key was just pressed this frame (was not pressed last frame) - edge detection
UP = 4  # Key was just released this frame (was pressed last frame) - edge detection


class Inputs(abc.ABC):
    def __init__(self):
        self.mouse_position = (0, 0)
        self.mouse_delta = (0, 0)
        self.scroll_delta = (0, 0)

    @staticmethod
    def from_render_canvas(canvas) -> Inputs:
        if canvas.__class__.__name__ == "GlfwRenderCanvas":
            return GLFWInputs(canvas)

        raise NotImplementedError(
            f"Input handling not implemented for this render canvas: {canvas.__class__.__name__}"
        )

    @abc.abstractmethod
    def is_key(self, key_code: int, action: int) -> bool:
        """
        Check if a key is in a specific state. Key code and action follows the
        GLFW convention with the addition of DOWN and UP actions for edge detection.
        """

    @abc.abstractmethod
    def is_mouse_button(self, button_code: int, action: int) -> bool:
        """
        Check if a mouse button is in a specific state. Button code and action follows the
        GLFW convention with the addition of DOWN and UP actions for edge detection.
        """

    @abc.abstractmethod
    def _update(self):
        """Update input state. Should be called once per frame."""


class GLFWInputs(Inputs):
    """
    GLFW input handler.
    """

    def __init__(self, canvas):
        try:
            import glfw

            self._glfw = glfw
        except ImportError:
            raise ImportError(
                "Desktop backend requires glfw.\nInstall with: pip install manifold-gfx[desktop]"
            )

        super().__init__()
        self._canvas = canvas

        self._last_frame_keys = set()
        self._last_frame_mouse_buttons = set()
        self._glfw.set_scroll_callback(self._canvas._window, self.scroll_callback)

    def scroll_callback(self, _window, xoffset, yoffset):
        self.scroll_delta = (xoffset, yoffset)

    def is_key(self, key_code: int, action: int) -> bool:
        act = self._glfw.get_key(self._canvas._window, key_code)
        was_down_last_frame = key_code in self._last_frame_keys
        if act == self._glfw.PRESS:
            self._last_frame_keys.add(key_code)

        if action < 3:  # GLFW actions
            return act == action
        if action == DOWN and act == self._glfw.PRESS and not was_down_last_frame:
            return True
        if action == UP and act == self._glfw.RELEASE and was_down_last_frame:
            return True

        return False

    def is_mouse_button(self, button_code: int, action: int) -> bool:
        act = self._glfw.get_mouse_button(self._canvas._window, button_code)
        was_down_last_frame = button_code in self._last_frame_mouse_buttons

        if act == self._glfw.PRESS:
            self._last_frame_mouse_buttons.add(button_code)

        if action < 3:  # GLFW actions
            return act == action
        if action == DOWN and act == self._glfw.PRESS and not was_down_last_frame:
            return True
        if action == UP and act == self._glfw.RELEASE and was_down_last_frame:
            return True

        return False

    def _update(self):
        self.scroll_delta = (0, 0)  # Reset scroll delta each frame
        self._glfw.poll_events()

        # Update mouse position and delta
        x, y = self._glfw.get_cursor_pos(self._canvas._window)
        self.mouse_delta = (x - self.mouse_position[0], y - self.mouse_position[1])
        self.mouse_position = (x, y)

        # Remove keys that are no longer pressed from last frame's set
        for k in list(self._last_frame_keys):
            if self._glfw.get_key(self._canvas._window, k) == self._glfw.RELEASE:
                self._last_frame_keys.remove(k)

        # Remove mouse buttons that are no longer pressed from last frame's set
        for b in list(self._last_frame_mouse_buttons):
            if self._glfw.get_mouse_button(self._canvas._window, b) == self._glfw.RELEASE:
                self._last_frame_mouse_buttons.remove(b)

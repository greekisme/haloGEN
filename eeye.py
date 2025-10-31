# emo_eyes.py
# EMO Robot Eye Animation System for Mira Voice Assistant (macOS Fixed)
# Install: pip install pygame

import pygame
import math
import time
import random
from enum import Enum
from dataclasses import dataclass
import os

# Fix for macOS: Force main thread
os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1'

class EyeState(Enum):
    """Eye expression states"""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HAPPY = "happy"
    CURIOUS = "curious"
    SLEEPY = "sleepy"
    ERROR = "error"

@dataclass
class EyeConfig:
    """Eye appearance configuration"""
    width: int = 900
    height: int = 650
    eye_width: int = 140  # Baymax eyes are smaller ovals
    eye_height: int = 140
    eye_spacing: int = 180
    bg_color: tuple = (245, 245, 245)  # Near-white Baymax face
    eye_color: tuple = (15, 15, 15)    # Black eyes/connector
    pupil_color: tuple = (0, 0, 0)     # Unused in Baymax mode
    baymax_mode: bool = True
    connector_width: int = 8
    # Waveform configuration
    waveform_height: int = 80
    waveform_margin_top: int = 40
    waveform_color: tuple = (20, 20, 20)
    waveform_bars: int = 48

class RobotEyes:
    """EMO-style robot eye animation system (macOS compatible)"""
    
    def __init__(self, config: EyeConfig = None):
        self.config = config or EyeConfig()
        self.running = True
        self.current_state = EyeState.IDLE
        self._next_state = None  # For thread-safe state changes
        
        # Eye parameters
        self.blink_progress = 0
        self.left_eye_offset_x = 0
        self.left_eye_offset_y = 0
        self.right_eye_offset_x = 0
        self.right_eye_offset_y = 0
        self.pupil_scale = 1.0
        self.animation_time = 0
        # Baymax mode timers
        self._next_blink = time.time() + 5.0
        self._blink_t = 0.0
        self._blinking = False

        # Waveform state
        self.waveform_mode = "off"  # off | listening | speaking
        self.waveform_envelope = None  # list[float] normalized 0..1
        self.waveform_start_time = None
        self.waveform_duration = 0.0
        
        # Subtitle state
        self.subtitle_text = ""  # legacy immediate text (kept for compatibility)
        self._subtitle_display = ""
        self._subtitle_change_time = 0.0
        self._subtitle_alpha = 0.0  # 0..1 for fade
        self._subtitle_fade_duration = 0.25  # seconds
        self._font = None

        # Live streaming subtitle state
        self.subtitle_full_text = ""
        self.subtitle_display_text = ""
        self.subtitle_next_update = 0
        self.subtitle_speed = 0.30  # seconds per word
        self._subtitle_words = []
        self._subtitle_word_index = 0
        
        # Captions toggle/button
        self.show_captions = True
        self._cc_rect = pygame.Rect(0, 0, 0, 0)

        # Micro-expression timers/state
        self.next_micro_event = time.time() + random.uniform(8, 12)
        self.micro_event_active = False
        self.micro_event_type = None
        self.micro_event_duration = 0.0
        self.micro_event_elapsed = 0.0
        self._micro_blink_boost = 0.0
        self._micro_eye_scale_left = 1.0
        self._micro_eye_scale_right = 1.0

        # Tap to Interrupt state/UI
        self.is_speaking = False
        self.tts_process = None  # external: set current playback process
        self.interrupt_button_rect = pygame.Rect(0, 0, 0, 0)  # set in _draw_interrupt_button
        self.interrupt_feedback = 0.0  # 0..1 flash progress
        self._interrupt_callback = None  # external function to stop TTS immediately

        # Themes
        self.themes = {
            "baymax": {
                "bg_color": (245, 245, 245),
                "eye_color": (15, 15, 15),
                "waveform_color": (20, 20, 20),
            },
            "dark": {
                "bg_color": (10, 10, 10),
                "eye_color": (230, 230, 230),
                "waveform_color": (180, 180, 180),
            },
            "aurora": {
                "bg_color": (12, 18, 30),
                "eye_color": (120, 200, 255),
                "waveform_color": (80, 160, 240),
            },
        }
        self._theme_order = ["baymax", "dark", "aurora"]
        self.current_theme = "baymax"
        self._apply_theme_colors()
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.width, self.config.height)
        )
        pygame.display.set_caption("Mira Eyes")
        self.clock = pygame.time.Clock()
        # Init font (system sans-serif)
        try:
            pygame.font.init()
            self._font = pygame.font.SysFont("Helvetica,Arial,Inter,San Francisco", 22)
        except Exception:
            self._font = pygame.font.Font(None, 22)

    def _apply_theme_colors(self):
        theme = self.themes.get(self.current_theme, self.themes["baymax"]) if hasattr(self, 'themes') else None
        if theme:
            # Update config colors in-place so all draw sites use them
            self.config.bg_color = theme.get("bg_color", self.config.bg_color)
            self.config.eye_color = theme.get("eye_color", self.config.eye_color)
            self.config.waveform_color = theme.get("waveform_color", self.config.waveform_color)

    def set_theme(self, name: str):
        if not name:
            return
        if name not in self.themes:
            return
        self.current_theme = name
        self._apply_theme_colors()

    def cycle_theme(self):
        try:
            idx = self._theme_order.index(self.current_theme)
        except Exception:
            idx = 0
        next_idx = (idx + 1) % len(self._theme_order)
        self.current_theme = self._theme_order[next_idx]
        self._apply_theme_colors()
        
    def set_state(self, state: EyeState):
        """Change the eye expression state (thread-safe)"""
        self._next_state = state
    
    def _apply_state_change(self):
        """Apply pending state change (call from main loop only)"""
        if self._next_state is not None:
            self.current_state = self._next_state
            self.animation_time = 0
            self._next_state = None
            # Auto-sync waveform mode to state
            try:
                if self.current_state == EyeState.SPEAKING:
                    self.set_waveform("speaking")
                elif self.current_state == EyeState.LISTENING:
                    self.set_waveform("listening")
                elif self.current_state == EyeState.THINKING:
                    self.set_waveform("thinking")
                else:
                    self.set_waveform("off")
            except Exception:
                # Fail-safe: never let UI crash due to waveform update
                pass
    
    def _draw_eye(self, surface, x, y, offset_x, offset_y, is_blinking):
        """Draw a single eye. Baymax mode uses a filled ellipse."""
        eye_w = self.config.eye_width
        eye_h = self.config.eye_height
        
        # Apply blink effect
        if is_blinking:
            eye_h = int(eye_h * (1 - self.blink_progress))
        
        if eye_h < 5:
            eye_h = 5  # Minimum height for closed eye line
        
        eye_rect = pygame.Rect(
            x + offset_x - eye_w // 2,
            y + offset_y - eye_h // 2,
            eye_w,
            eye_h
        )
        if self.config.baymax_mode:
            pygame.draw.ellipse(surface, self.config.eye_color, eye_rect)
        else:
            pygame.draw.rect(
                surface,
                self.config.eye_color,
                eye_rect,
                border_radius=min(eye_w, eye_h) // 2
            )
        
        # Draw pupil/highlight only in non-Baymax mode
        if not self.config.baymax_mode and eye_h > 30:
            pupil_w = int(eye_w * 0.4 * self.pupil_scale)
            pupil_h = int(eye_h * 0.4 * self.pupil_scale)
            pupil_rect = pygame.Rect(
                x + offset_x - pupil_w // 2,
                y + offset_y - pupil_h // 2,
                pupil_w,
                pupil_h
            )
            pygame.draw.ellipse(surface, self.config.pupil_color, pupil_rect)
    
    def _update_idle(self, dt):
        """Idle animation - occasional blinks and subtle movement"""
        self.animation_time += dt
        
        # Random blinks
        if self.animation_time > 3.0:
            self.blink_progress = min(1.0, self.blink_progress + dt * 8)
            if self.blink_progress >= 1.0:
                self.animation_time = 0
                self.blink_progress = 0
        
        # Subtle breathing motion
        breath = math.sin(self.animation_time * 1.5) * 5
        self.left_eye_offset_y = breath
        self.right_eye_offset_y = breath
        self.pupil_scale = 1.0 + math.sin(self.animation_time * 1.5) * 0.05
    
    def _update_listening(self, dt):
        """Listening animation - eyes focused and alert"""
        self.animation_time += dt
        
        # Attentive pulsing
        pulse = math.sin(self.animation_time * 4) * 0.1
        self.pupil_scale = 1.1 + pulse
        
        # Quick blinks
        if int(self.animation_time * 2) % 4 == 0:
            self.blink_progress = min(1.0, self.blink_progress + dt * 10)
        else:
            self.blink_progress = max(0, self.blink_progress - dt * 10)
    
    def _update_thinking(self, dt):
        """Thinking animation - eyes looking around"""
        self.animation_time += dt
        
        # Eyes move in a figure-8 pattern
        t = self.animation_time * 1.5
        self.left_eye_offset_x = math.sin(t) * 20
        self.left_eye_offset_y = math.sin(t * 2) * 15
        self.right_eye_offset_x = math.sin(t + 0.2) * 20
        self.right_eye_offset_y = math.sin(t * 2 + 0.2) * 15
        self.pupil_scale = 0.9
    
    def _update_speaking(self, dt):
        """Speaking animation - synchronized blinking"""
        self.animation_time += dt
        
        # Rhythmic blinking
        blink_cycle = (self.animation_time * 3) % 2
        if blink_cycle < 0.2:
            self.blink_progress = blink_cycle * 5
        else:
            self.blink_progress = 0
        
        # Slight vertical movement
        self.left_eye_offset_y = math.sin(self.animation_time * 5) * 3
        self.right_eye_offset_y = math.sin(self.animation_time * 5) * 3
        self.pupil_scale = 1.0
    
    def _update_happy(self, dt):
        """Happy animation - wide eyes with bounce"""
        self.animation_time += dt
        
        # Bouncy movement
        bounce = abs(math.sin(self.animation_time * 4)) * 15
        self.left_eye_offset_y = -bounce
        self.right_eye_offset_y = -bounce
        self.pupil_scale = 1.2
        
        # Happy blinks
        if int(self.animation_time * 2) % 6 == 0:
            self.blink_progress = min(0.5, self.blink_progress + dt * 8)
        else:
            self.blink_progress = max(0, self.blink_progress - dt * 8)
    
    def _update_curious(self, dt):
        """Curious animation - one eye squinted, looking around"""
        self.animation_time += dt
        
        # Asymmetric look
        look_angle = math.sin(self.animation_time * 2) * 30
        self.left_eye_offset_x = look_angle
        self.right_eye_offset_x = look_angle * 0.7
        self.pupil_scale = 1.1
    
    def _update_sleepy(self, dt):
        """Sleepy animation - slow blinks, droopy eyes"""
        self.animation_time += dt
        
        # Slow drooping blinks
        blink_cycle = math.sin(self.animation_time * 0.8)
        self.blink_progress = max(0, blink_cycle * 0.7)
        
        # Droopy effect
        self.left_eye_offset_y = 10
        self.right_eye_offset_y = 10
        self.pupil_scale = 0.8
    
    def _update_error(self, dt):
        """Error animation - rapid blinking"""
        self.animation_time += dt
        
        # Rapid warning blinks
        if int(self.animation_time * 8) % 2 == 0:
            self.blink_progress = 1.0
        else:
            self.blink_progress = 0
        
        # Shake effect
        self.left_eye_offset_x = math.sin(self.animation_time * 20) * 10
        self.right_eye_offset_x = math.sin(self.animation_time * 20 + math.pi) * 10
    
    def _update(self, dt):
        """Update eye animation based on current state"""
        # Reset offsets
        self.left_eye_offset_x = 0
        self.left_eye_offset_y = 0
        self.right_eye_offset_x = 0
        self.right_eye_offset_y = 0
        if self.config.baymax_mode:
            # Baymax: gentle breathing and occasional quick blink
            self.animation_time += dt
            # Gentle vertical breathing (2px amplitude)
            breath = math.sin(self.animation_time * 1.2) * 2
            self.left_eye_offset_y = breath
            self.right_eye_offset_y = breath
            # Tiny horizontal drift (1px amplitude)
            drift = math.sin(self.animation_time * 0.6) * 1
            self.left_eye_offset_x = drift
            self.right_eye_offset_x = drift
            # Timed blink every 3.5–7s
            now = time.time()
            if not self._blinking and now >= self._next_blink:
                self._blinking = True
                self._blink_t = 0.0
                self._next_blink = now + 3.5 + (random.random() * 3.5)
            if self._blinking:
                self._blink_t += dt
                # 0.22s total blink using a sine ease
                total = 0.22
                phase = min(1.0, self._blink_t / total)
                self.blink_progress = math.sin(phase * math.pi)
                if self._blink_t >= total:
                    self._blinking = False
                    self.blink_progress = 0.0
            else:
                self.blink_progress = 0.0
            # No pupil in Baymax mode
            self.pupil_scale = 1.0
            # Micro-expressions overlay (calm states only)
            calm_states = {EyeState.IDLE, EyeState.LISTENING, EyeState.THINKING}
            self._update_micro_expressions(dt, self.current_state in calm_states)
            return
        
        # Update based on state
        state_updates = {
            EyeState.IDLE: self._update_idle,
            EyeState.LISTENING: self._update_listening,
            EyeState.THINKING: self._update_thinking,
            EyeState.SPEAKING: self._update_speaking,
            EyeState.HAPPY: self._update_happy,
            EyeState.CURIOUS: self._update_curious,
            EyeState.SLEEPY: self._update_sleepy,
            EyeState.ERROR: self._update_error,
        }
        
        update_func = state_updates.get(self.current_state, self._update_idle)
        update_func(dt)
        # Micro-expressions overlay (calm states only)
        calm_states = {EyeState.IDLE, EyeState.LISTENING, EyeState.THINKING}
        self._update_micro_expressions(dt, self.current_state in calm_states)

    def _update_micro_expressions(self, dt, is_calm: bool):
        # Reset horizontal/vertical offsets always (no movement)
        self.left_eye_offset_x = 0
        self.right_eye_offset_x = 0
        self.left_eye_offset_y = 0
        self.right_eye_offset_y = 0

        # Default scales
        self._micro_eye_scale_left = 1.0
        self._micro_eye_scale_right = 1.0
        self._micro_blink_boost = 0.0

        now = time.time()
        if not is_calm:
            # Not calm; ensure event cleared and schedule later
            self.micro_event_active = False
            self.micro_event_type = None
            self.micro_event_elapsed = 0.0
            if now > self.next_micro_event:
                self.next_micro_event = now + random.uniform(8, 12)
            return

        if not self.micro_event_active:
            if now >= self.next_micro_event:
                self.micro_event_active = True
                self.micro_event_type = random.choice(["squint_left", "squint_right", "blink_soft", "breath_mini"])
                self.micro_event_duration = random.uniform(0.1, 0.3)
                self.micro_event_elapsed = 0.0
        
        if self.micro_event_active:
            self.micro_event_elapsed += dt
            p = max(0.0, min(1.0, self.micro_event_elapsed / max(0.001, self.micro_event_duration)))
            # Smooth in-out curve
            s = math.sin(p * math.pi)
            if self.micro_event_type == "squint_left":
                self._micro_eye_scale_left = max(0.8, 1.0 - 0.12 * s)
            elif self.micro_event_type == "squint_right":
                self._micro_eye_scale_right = max(0.8, 1.0 - 0.12 * s)
            elif self.micro_event_type == "breath_mini":
                k = 0.04 * s
                self._micro_eye_scale_left = 1.0 - k
                self._micro_eye_scale_right = 1.0 - k
            elif self.micro_event_type == "blink_soft":
                self._micro_blink_boost = 0.5 * s  # subtle, max 0.5

            if self.micro_event_elapsed >= self.micro_event_duration:
                # Finish and schedule next
                self.micro_event_active = False
                self.micro_event_type = None
                self.micro_event_elapsed = 0.0
                self.next_micro_event = now + random.uniform(8, 12)
    
    def _draw(self):
        """Draw the eyes and waveform with supersampling for smoothness"""
        w, h = self.config.width, self.config.height
        super_res = 2  # 2x supersampling
        w_s, h_s = w * super_res, h * super_res
        temp = pygame.Surface((w_s, h_s))

        # background
        temp.fill(self.config.bg_color)

        # positions in super-res space
        center_y = (self.config.height // 2) * super_res
        eye_distance = (self.config.eye_width + self.config.eye_spacing) * super_res
        left_x = (self.config.width * super_res) // 2 - eye_distance // 2
        right_x = (self.config.width * super_res) // 2 + eye_distance // 2

        # connector line
        pygame.draw.line(
            temp,
            self.config.eye_color,
            (left_x, center_y),
            (right_x, center_y),
            self.config.connector_width * super_res
        )

        # eyes – fixed position, blink = squeeze
        # Include micro blink boost (subtle extra closing)
        blink = max(0.05, 1 - min(1.0, self.blink_progress + self._micro_blink_boost))
        eye_w = self.config.eye_width * super_res
        eye_h_base = int(self.config.eye_height * super_res * blink)

        for i, x in enumerate([left_x, right_x]):
            # Apply micro eye height scaling per eye
            scale = self._micro_eye_scale_left if i == 0 else self._micro_eye_scale_right
            eye_h = max(5, int(eye_h_base * scale))
            rect = pygame.Rect(
                x - eye_w // 2,
                center_y - eye_h // 2,
                eye_w,
                eye_h
            )
            pygame.draw.ellipse(temp, self.config.eye_color, rect)
            # Draw moving glass-like highlight overlay
            self._draw_eye_highlight(temp, rect, super_res)

        # draw waveform below, aligned between eye centers
        self._draw_waveform(
            temp,
            int(center_y + (self.config.eye_height + self.config.waveform_margin_top) * super_res),
            super_res,
            left_x,
            right_x,
        )

        # downsample smoothly to screen
        smooth = pygame.transform.smoothscale(temp, (w, h))
        self.screen.blit(smooth, (0, 0))
        # Draw subtitles on final surface space (no supersampling needed)
        self._draw_subtitle()
        # Draw CC toggle button in window space
        self._draw_cc_button()
        # Draw interrupt button
        self._draw_interrupt_button()
        pygame.display.flip()



    def _current_speaking_amp(self):
        if not self.waveform_envelope or self.waveform_start_time is None or self.waveform_duration <= 0:
            return 0.0
        t = time.time() - self.waveform_start_time
        if t < 0:
            return 0.0
        if t >= self.waveform_duration:
            return 0.0
        idx = int((t / self.waveform_duration) * (len(self.waveform_envelope) - 1))
        idx = max(0, min(idx, len(self.waveform_envelope) - 1))
        return float(self.waveform_envelope[idx])

    def _draw_waveform(self, surface, base_y, super_res=1, left_x=None, right_x=None):
        """ChatGPT / Perplexity style waveform animation, spanned between eye centers"""
        bars = 32
        gap = 6 * super_res
        # Desired span: between eye centers (connector line endpoints)
        if left_x is None or right_x is None:
            # Fallback to full-width centering
            width = self.config.width * super_res
            desired_total = width
            span_left = 0
        else:
            desired_total = max(0, right_x - left_x)
            span_left = left_x

        # Compute bar width to fit desired span with fixed gap
        bar_width = max(2 * super_res, (desired_total - gap * (bars - 1)) // bars if desired_total > 0 else 6 * super_res)
        actual_total = bars * bar_width + gap * (bars - 1)
        # Center within the span if there's rounding leftover
        start_x = int(span_left + max(0, (desired_total - actual_total) // 2))
        max_h = self.config.waveform_height * super_res

        t = time.time() * 3
        amps = []
        if self.waveform_mode == "speaking":
            # dynamic pulse based on envelope
            amp = self._current_speaking_amp() or 0.4
            for i in range(bars):
                phase = (i / bars) * math.pi * 2
                val = amp * (0.6 + 0.4 * math.sin(t + phase))
                amps.append(val)
        elif self.waveform_mode == "listening":
            for i in range(bars):
                phase = (i / bars) * math.pi * 2
                val = 0.2 + 0.1 * math.sin(t * 0.5 + phase)
                amps.append(val)
        elif self.waveform_mode == "thinking":
            for i in range(bars):
                phase = (i / bars) * math.pi * 2
                val = 0.08 + 0.04 * math.sin(t * 0.25 + phase)
                amps.append(val)
        else:
            return

        color = (30, 30, 30)
        for i, a in enumerate(amps):
            h = int(max(1, a * max_h))
            x = start_x + i * (bar_width + gap)
            y = base_y - h // 2
            rect = pygame.Rect(x, y, bar_width, h)
            pygame.draw.rect(surface, color, rect, border_radius=bar_width // 2)

    def _draw_cc_button(self):
        # Small rounded rectangle with 'CC' label at top-right
        try:
            font = self._font or pygame.font.Font(None, 22)
        except Exception:
            return
        pad = 12
        label = "CC"
        text_surf = font.render(label, True, (255, 255, 255))
        tw, th = text_surf.get_size()
        bw, bh = tw + 14, th + 8
        x = self.config.width - pad - bw
        y = pad
        self._cc_rect = pygame.Rect(x, y, bw, bh)
        # Visual state
        bg_off = (200, 200, 200)
        bg_on = (90, 90, 90)
        bg = bg_on if self.show_captions else bg_off
        pygame.draw.rect(self.screen, bg, self._cc_rect, border_radius=8)
        # Center the label
        tr = text_surf.get_rect(center=self._cc_rect.center)
        self.screen.blit(text_surf, tr)

    def _draw_interrupt_button(self):
        # Short labels: "tap" when speaking, "list" when listening
        label = "tap" if self.is_speaking else "list"
        # Button colors
        bg = (40, 40, 40) if self.is_speaking else (180, 180, 180)
        fg = (255, 255, 255) if self.is_speaking else (30, 30, 30)
        # Flash feedback overlay (white flash decay)
        if self.interrupt_feedback > 0.0:
            self.interrupt_feedback = max(0.0, self.interrupt_feedback - 0.08)
        # Position next to CC button (left of it)
        try:
            font = self._font or pygame.font.Font(None, 22)
        except Exception:
            return
        text_surf = font.render(label, True, fg)
        tw, th = text_surf.get_size()
        bw, bh = tw + 14, th + 8
        pad = 12
        # Place to the left of CC button
        x = self.config.width - pad - self._cc_rect.width - pad - bw
        y = pad
        self.interrupt_button_rect = pygame.Rect(x, y, bw, bh)
        pygame.draw.rect(self.screen, bg, self.interrupt_button_rect, border_radius=8)
        # Center the label
        tr = text_surf.get_rect(center=self.interrupt_button_rect.center)
        self.screen.blit(text_surf, tr)
        if self.interrupt_feedback > 0.0:
            flash = pygame.Surface((self.interrupt_button_rect.width, self.interrupt_button_rect.height), pygame.SRCALPHA)
            alpha = int(255 * min(1.0, self.interrupt_feedback))
            flash.fill((255, 255, 255, alpha))
            self.screen.blit(flash, self.interrupt_button_rect)

    def _draw_eye_highlight(self, surface, eye_rect: pygame.Rect, super_res: int):
        # Semi-transparent diagonal highlight that moves slightly over time
        t = time.time()
        # Create a small overlay surface matching eye bounds
        overlay = pygame.Surface((eye_rect.width, eye_rect.height), pygame.SRCALPHA)
        # Build a soft gradient by stacking several translucent ellipses
        # Narrow highlight strip dimensions and position drift
        drift_x = int(math.sin(t * 0.6) * (eye_rect.width * 0.04))
        drift_y = int(math.cos(t * 0.5) * (eye_rect.height * 0.03))
        hw = int(eye_rect.width * 0.55)
        hh = int(max(1, eye_rect.height * 0.24))
        cx = eye_rect.width // 2 + drift_x
        cy = eye_rect.height // 2 - int(eye_rect.height * 0.18) + drift_y
        # Draw concentric ellipses for a faux gradient
        steps = 6
        for s in range(steps):
            k = 1.0 - (s / steps)
            w = max(2, int(hw * k))
            h = max(2, int(hh * k))
            alpha = int(35 * k)  # subtle
            r = pygame.Rect(0, 0, w, h)
            r.center = (cx, cy)
            pygame.draw.ellipse(overlay, (255, 255, 255, alpha), r)
        # Rotate overlay slightly to create diagonal look
        rotated = pygame.transform.rotate(overlay, -28)
        # Compute blit position so rotated center aligns near eye center
        dest = rotated.get_rect(center=eye_rect.center)
        surface.blit(rotated, dest)

    def _draw_subtitle(self):
        if self._font is None:
            return
        if not self.show_captions:
            return
        # Update streaming state (word-by-word)
        now = time.time()
        if self.subtitle_full_text and self.subtitle_next_update and now >= self.subtitle_next_update:
            if self._subtitle_word_index < len(self._subtitle_words):
                # Reveal next word
                if self.subtitle_display_text:
                    self.subtitle_display_text += " " + self._subtitle_words[self._subtitle_word_index]
                else:
                    self.subtitle_display_text = self._subtitle_words[self._subtitle_word_index]
                self._subtitle_word_index += 1
                self.subtitle_next_update = now + max(0.01, float(self.subtitle_speed))
                # Start or continue fade-in when streaming begins
                if self._subtitle_alpha <= 0.0:
                    self._subtitle_change_time = now
            else:
                # Completed; stop further scheduling
                self.subtitle_next_update = 0

        # Update fade-in progression
        if self._subtitle_alpha < 1.0:
            elapsed = now - self._subtitle_change_time
            if elapsed >= 0:
                self._subtitle_alpha = min(1.0, elapsed / max(0.001, self._subtitle_fade_duration))

        # Choose current text to render: prefer streaming display, fallback to legacy
        current_text = self.subtitle_display_text or (self.subtitle_text or "")
        if not current_text:
            return

        # Wrap text to 80% width and keep only last 1–2 lines
        w, h = self.config.width, self.config.height
        max_width = int(w * 0.8)
        words = current_text.split()
        lines = []
        line = ""
        append = lines.append
        size = self._font.size
        for word in words:
            test = word if not line else f"{line} {word}"
            if size(test)[0] <= max_width:
                line = test
            else:
                if line:
                    append(line)
                line = word
        if line:
            append(line)
        # Keep last 2 lines
        lines = lines[-2:]

        # Render centered near bottom (around 88% height), with clean spacing
        color = self.config.eye_color
        alpha_value = int(255 * self._subtitle_alpha)
        line_height = self._font.get_linesize()
        total_height = len(lines) * line_height + (len(lines) - 1) * int(line_height * 0.2)
        base_y = int(h * 0.88) - total_height // 2
        y = base_y
        for ln in lines:
            try:
                surf = self._font.render(ln, True, color).convert_alpha()
                surf.set_alpha(alpha_value)
            except Exception:
                continue
            rect = surf.get_rect()
            rect.centerx = w // 2
            rect.y = y
            self.screen.blit(surf, rect)
            y += line_height + int(line_height * 0.2)

    
    def update(self):
        """Single frame update - call this repeatedly from main thread"""
        # Apply any pending state changes
        self._apply_state_change()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.show_captions = not self.show_captions
                elif event.key == pygame.K_t:
                    self.cycle_theme()
                elif event.key == pygame.K_SPACE:
                    self._on_interrupt_trigger()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if self._cc_rect.collidepoint(mx, my):
                        self.show_captions = not self.show_captions
                    if self.interrupt_button_rect.collidepoint(mx, my):
                        self._on_interrupt_trigger()
        
        # Update animation
        dt = self.clock.tick(60) / 1000.0  # 60 FPS
        self._update(dt)
        self._draw()
        
        return True

    def _on_interrupt_trigger(self):
        # Flash
        self.interrupt_feedback = 1.0
        # Stop external TTS if provided (barge-in)
        try:
            if callable(self._interrupt_callback):
                self._interrupt_callback()
            elif self.tts_process is not None:
                # Best-effort terminate
                try:
                    if hasattr(self.tts_process, 'terminate'):
                        self.tts_process.terminate()
                except Exception:
                    pass
        except Exception:
            pass
        # Switch to listening visuals
        self.is_speaking = False
        self.set_state(EyeState.LISTENING)
        self.set_waveform("listening")
    
    def run(self):
        """Run the animation loop (blocking - must be called from main thread)"""
        self.running = True
        while self.running:
            if not self.update():
                break
    
    def quit(self):
        """Cleanup"""
        pygame.quit()

    def set_waveform(self, mode: str, envelope=None, duration: float | None = None):
        """Control the waveform renderer.

        mode: "off" | "listening" | "speaking"
        envelope: list of floats (0..1) for speaking mode
        duration: total seconds for the envelope playback
        """
        if mode not in ("off", "listening", "speaking", "thinking"):
            mode = "off"
        self.waveform_mode = mode
        if mode == "speaking" and envelope and duration and duration > 0:
            self.waveform_envelope = [max(0.0, min(1.0, float(v))) for v in envelope]
            self.waveform_duration = float(duration)
            self.waveform_start_time = time.time()
        else:
            self.waveform_envelope = None
            self.waveform_duration = 0.0
            self.waveform_start_time = None

    def set_subtitle(self, text: str):
        """Start streaming subtitle text word-by-word, with fade-in."""
        if text is None:
            text = ""
        # Streaming setup
        self.subtitle_full_text = str(text)
        self.subtitle_display_text = ""
        self._subtitle_words = [w for w in self.subtitle_full_text.split()]
        self._subtitle_word_index = 0
        now = time.time()
        self.subtitle_next_update = now + max(0.01, float(self.subtitle_speed)) if self._subtitle_words else 0
        # Fade from 0
        self._subtitle_alpha = 0.0
        self._subtitle_change_time = now
        # Maintain legacy field for compatibility, not used for rendering when streaming is active
        self.subtitle_text = self.subtitle_full_text


# Demo/Test code
if __name__ == "__main__":
    print("EMO Robot Eyes - Running")
    print("Press close button or Ctrl+C to exit")
    print("-" * 40)
    
    eyes = RobotEyes()
    eyes.set_state(EyeState.IDLE)
    
    try:
        # Just run the eyes indefinitely in IDLE state
        while eyes.running:
            if not eyes.update():
                break
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        eyes.quit()
        print("Demo complete!")
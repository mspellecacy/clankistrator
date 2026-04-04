# Spinning ASCII Cube Design Document

## 1. Overview
This document outlines the design for a simple spinning 3D cube rendered in ASCII art, using Python. The application is intended to run in a standard terminal and is structured to be easily understandable and modifiable by AI coding agents and orchestrators.

## 2. Goals
*   **Simplicity:** The code should be straightforward, relying on standard Python libraries only.
*   **AI-Friendly:** Clear separation of concerns (math, rendering, main loop) to allow AI agents to easily swap out components or add features.
*   **Terminal Output:** Render the 3D cube as 2D ASCII art directly in the console.
*   **Smooth Animation:** Provide a continuous spinning effect.

## 3. Technology Stack
*   **Language:** Python 3.x
*   **Libraries:**
    *   `math`: For trigonometric functions (sin, cos) used in 3D rotation.
    *   `time`: For controlling the frame rate (sleep).
    *   `os` / `sys`: For terminal clearing and output manipulation.

## 4. Architecture / Components

### 4.1. The 3D Math Engine (Cube Representation & Rotation)
*   **Data Structure:** The cube will be defined by its 8 vertices in 3D space (x, y, z).
    *   Vertices can be represented as a list of lists or tuples: e.g., `[[-1, -1, -1], [1, -1, -1], ...]`.
*   **Rotation Logic:**
    *   Apply rotation matrices for the X, Y, and Z axes.
    *   State variables: `angle_x`, `angle_y`, `angle_z`.
    *   Update function: Takes the original vertices, applies current rotation angles, and returns transformed 3D coordinates.

### 4.2. The Projection Engine (3D to 2D)
*   **Perspective Projection:** Convert the rotated 3D coordinates (x, y, z) into 2D screen coordinates (x', y').
    *   Formula: `x' = (x * FOV) / (z + Distance)` and `y' = (y * FOV) / (z + Distance)`.
    *   `Distance` ensures the cube doesn't clip through the "camera".
*   **Screen Mapping:** Map the calculated 2D coordinates to integer coordinates representing character positions on the terminal grid.

### 4.3. The Rendering Engine (ASCII Drawing)
*   **Frame Buffer:** A 2D array (or a 1D array mapped to 2D) representing the screen characters. Initially filled with spaces (' ').
*   **Z-Buffer (Optional but recommended for solid look):** A parallel array storing the depth (z-value) of each rendered pixel to handle occlusion. If simplicity is paramount, a wireframe approach (drawing lines between projected vertices) might skip this, but a solid face approach requires it. For a simple wireframe, just drawing lines is enough.
*   **Line Drawing (Bresenham's Algorithm):** An algorithm to draw straight lines between the projected 2D vertices to form the edges of the cube.
*   **Output:** A function to print the frame buffer to the console, preceded by a terminal clear command (or moving the cursor to the top-left to avoid flickering).

### 4.4. The Main Loop (Orchestrator)
*   **Initialization:** Set up the initial cube vertices, screen size, and starting angles.
*   **Execution Loop:**
    1.  Clear the frame buffer.
    2.  Update rotation angles (increment by a small step).
    3.  Rotate all 3D vertices.
    4.  Project 3D vertices to 2D.
    5.  Draw lines between projected vertices into the frame buffer.
    6.  Print the frame buffer to the console.
    7.  Wait for a short duration (e.g., 0.03 seconds) to maintain a consistent frame rate.

## 5. Development Phases for AI Agents

*   **Phase 1: Basic Math & Projection.** Implement the 3D vertex definitions, rotation functions, and perspective projection. Create simple tests to verify the math.
*   **Phase 2: Wireframe Rendering.** Implement a line-drawing algorithm and the 2D frame buffer. Connect the projected points to draw the 12 edges of the cube.
*   **Phase 3: Animation Loop.** Tie Phase 1 and 2 together in a continuous `while True` loop, updating angles and clearing the screen.

## 6. Future Enhancements (For advanced AI tasks)
*   Solid faces with lighting/shading (using dot products for normals).
*   Handling dynamic terminal resizing.
*   Adding user input to control rotation speed or direction.
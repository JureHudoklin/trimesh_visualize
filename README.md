# Trimesh Visualize
A wrapper for trimesh that simplifies visualization.

## 1. Installation

1. Install the requirements by running

```bash 
pip install -r requirements.txt
``` 

2. Install the package with pip
```bash 
pip install .
``` 
## 2. Usage

- Import the package and create a "Scene"

    ```python
    from trimeshVisualize import Scene

    scene = Scene()
    ```

- Add objects, points, vectors, points clouds, ... to the scene 
    - Colors are specified as [R, G, B, transparency]

    ```python
    scene.plot_mesh(mesh, color = [255, 0, 0, 255], id = "obj_1")
    ```

- Remove objects from the scene

    ```python
    scene.remove_feature("obj_1")
    ```

- Plot the scene

    ```python
    _ = scene.display()
    ```

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import shapely
import random
import string
import os
import io
import PIL.Image as Image


def id_generator(size=6, chars=list('0123456789')):
    return ''.join(np.random.choice(chars, size=size))

 
def id_generator(base = ""):
    """
    Generates a random ID in the form of a string
    --------------
    Keyword Arguments:
        base -- a string to be used as the base of the ID
    Returns
    --------------
    id : string
        A random string of ascii characters and digits
    """
    id = base.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(8))
    return id

class Scene():
    def __init__(self):
        self.meshes = {}

    def display(self, my_scene=None, clear_meshes=True, add_lights=True):
        """
        Plots all the meshes that were cached.
        --------------
        Keyword Arguments:
            - my_scene : trimesh.Scene {defult: None}
                If provided, the scene is updated with the meshes
            - clear_meshes : Bool
                If true resets all the cashed meshes after plotting
            - add_lights : Bool
                If true adds lights to the scene
        Return:
        --------------
            - new_scene : trimesh.Scene
                Updated trimesh scene with the objects/features
        """
        # Create the scene
        if my_scene == None:
            my_scene = trimesh.scene.scene.Scene(self.meshes)
        else:
            for mesh in self.meshes.values():
                my_scene.add_geometry(mesh)

        # Add a specific light
        if add_lights == True:
            light_tr_MX = trimesh.transformations.translation_matrix(
                np.array([0, 100, 10]))
            lights = trimesh.scene.lighting.DirectionalLight(
                name="light_1", color=[100, 100, 100, 255], intensity=5.0, radius=1)
            my_scene.graph[lights.name] = light_tr_MX
            my_scene.lights = [lights]

        # Plot scene
        my_scene.show()

        if clear_meshes == True:
            self.meshes = []

        return my_scene

    def plot_point(self, point, color=[255, 0, 0, 255], radius=1, id = None):
        """
        Plots a point 
        --------------
        point : np.array(3,)
            3D coordinates of the point
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        radius : float
            Radius of the displayed point
        id : string {defult:point_...}
            ID of the point
        
        Return
        --------------
        id : str
        """
        if id is None:
            id = id_generator("point_")
        mesh_point = trimesh.primitives.Sphere(radius=radius)
        mesh_point.apply_translation(point)
        mesh_point.visual.face_colors = color
        self.meshes[id] = mesh_point

        return id

    def plot_point_multiple(self, points, color=[255, 0, 0, 255], radius=0.5):
        """
        Plots points
        --------------
        points : np.array(n,3)
            3D coordinates of the point
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        radius : float
            Radius of the displayed point
        
        Return:
        --------------
        None : None
        """
        for point in points:
            id = id_generator("point_")
            mesh_point = trimesh.primitives.Sphere(radius=radius)
            mesh_point.apply_translation(point)
            mesh_point.visual.face_colors = color
            self.meshes[id] = mesh_point


    def plot_point_cloud(self, pc, color=[255, 0, 0, 255], radius=0.5, id = None):
        """
        Plots a point cloud
        --------------
        pc : np.array(n,3)
            Point cloud
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        radius : float
            Radius of the displayed point
        id : string {defult:pc_...}
            ID of the point cloud
        
        Return:
        --------------
        id : str
        """
        if id is None:
            id = id_generator("pc_")
        pc_mesh = trimesh.points.PointCloud(pc, colors=color, radius=radius)
        self.meshes[id] = pc_mesh

        return id

    def plot_cone(self, height, radius,
                tf = None, 
                point = None,
                direction = None,
                from_base = True,
                color = [255, 0, 0, 255],
                id = None):
        """
        Plots a cone.
        --------------
        height : int
            Height of the cone.
        radius : int
            Radius of the cone base
        tf : np.array() [4, 4]
            4x4 tf matrix. 
        point : np.array() [3]
            Location of the cone top. {Can be provided instead of tf.}
        direction : np.array() [3]
            Vector pointing from cone top to the base. {Can be provided instead of tf.}
        from_base : bool {defult: True}
            If true, the cone is plotted from the base. Otherwise, it is plotted from the top.
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        id : string {defult:cone_...}
            ID of the cone cloud
        
        Return:
        --------------
        id : str
        """

        id = id_generator("cone_")
        cone = trimesh.creation.cone(radius=radius, height=height)
        cone.visual.face_colors = color
        if from_base == False:
            cone.apply_translation([0, 0, -height])
            cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

        if tf is not None:
            cone_tf = tf
                
            
        elif point is not None and direction is not None:
            tf = trimesh.geometry.align_vectors(np.array([0, 0, 1]), direction)
            tf[:3, 3] = point
            cone_tf = tf
            
        else:
            raise ValueError("Transformation matrix or point and direction must be provided.")
            
        cone.apply_transform(cone_tf)
        self.meshes[id] = cone

        return id

    def plot_mesh(self, mesh, color=[150, 150, 150, 255], **kwargs):
        """
        Plots the mesh given a trimesh mesh
        --------------
        Arguments:
            mesh : trimesh.Trimesh
                Mesh to be plotted
            
        Keyword Arguments:
            color : array(4,)
                    Color of the point: [R,G,B, Intensity]
            id : string {defult:mesh_...}
                ID of the mesh
            file_loc : string {defult:None}
                If provided mesh gets loaded from file
            units : list {defult: None}
                If specified mesh units is changed to units[1] from units[0]

        Return:
        --------------
            None : None
        """
        if "file_loc" in kwargs:
            mesh = trimesh.load(kwargs["file_loc"])
        if "units" in kwargs:   # Convert mesh to correct units
            mesh.units = kwargs["units"][0]
            mesh.convert_units(kwargs["units"][1])

        if "id" in kwargs:
            id = kwargs["id"]
        else:
            id = id_generator("mesh_")
        
        # Unmerge so viewer doesn't smooth
        mesh.unmerge_vertices()
        # Assign color
        mesh.visual.face_colors = color
        # Append to meshes
        self.meshes[id] = mesh

    def plot_vector(self, p1, p2, color=[0, 0, 0, 255], radius_cyl=0.6, arrow=True, id = None):
        """
        Plats a vector from point 1 to point 2
        --------------
        Arguments:
            p1 : np.array(3,)
                3D coordinates of point 1
            p2 : np.array(3,)
                3D coordinates of point 2
        Keyword Arguments:
            color : array(4,)
                Color of the point: [R,G,B, Intensity]
            radius_cyl : float
                The size of the arrow
            arrow : Bool
                If true arrow is displayed at the end of the vector, otherwise just a line
            id : string {defult:vector_...}
        
        Return:
        --------------
        None : None
        """
        vector = p2-p1
        length = np.linalg.norm(p2-p1)
        if length < 0.0001:
            return
        if arrow:
            height_cylinder = length * 8/10
            height_cone = length * 2/10
        else:
            height_cylinder = length
            height_cone = length * 2/10
        rot_MX = trimesh.geometry.align_vectors(np.array([0, 0, 1]), vector)
        tr_MX = trimesh.transformations.translation_matrix(p1)

        # Create Cylinder
        cylinder = trimesh.primitives.Cylinder(
            radius=radius_cyl, height=height_cylinder, sections=3)
        cylinder.visual.face_colors = color
        cylinder.apply_translation(np.array([0, 0, height_cylinder/2]))
        cylinder.apply_transform(rot_MX)
        cylinder.apply_transform(tr_MX)

        # Create Cone
        cone = trimesh.creation.cone(radius=radius_cyl*1.5, height=height_cone)
        cone.visual.face_colors = color
        tr_cone = trimesh.transformations.translation_matrix(
            np.array([0, 0, height_cylinder]))
        cone.apply_transform(tr_cone)
        cone.apply_transform(rot_MX)
        cone.apply_transform(tr_MX)

        if id is None:
            id = id_generator("vector_") 

        if arrow:
            concateneted_vector = trimesh.util.concatenate([cylinder, cone])
            self.meshes[id] = concateneted_vector
        else:
            self.meshes[id] = cylinder


    def plot_grasp(self, grasp_tf, score=1, units="millimeters", color = [0, 0, 255, 255]):
        """
        Given grasp transform matrices and their score, plots the grasp
        --------------
        Arguments:
            grasp_tf : np.array(4,4)
                Transformation matrix
            score : np.array(n)
                Scores of individual grasps
        Return:
        --------------
            None : None
        """

        if not isinstance(grasp_tf, np.ndarray):
            try:
                grasp_tf = np.array(grasp_tf)
            except:
                raise ValueError("grasp_tf must be a numpy array")

        if units == "millimeters":
            length_coef = 0.01
        else:
            length_coef = 10
            
        if grasp_tf.ndim == 2:
            if round(score, 4) == 0:
                return None
            grasp_point = np.array([0, 0, 0])
            grasp_dir = np.array([0, 0, length_coef*3*score])
            points_transformed = trimesh.transform_points(
                [grasp_point, grasp_dir], grasp_tf)
            grasp_point = np.array(points_transformed[0])
            grasp_dir = np.array(points_transformed[1])
            id = self.plot_vector(grasp_point, grasp_dir,
                             radius_cyl=length_coef/10, arrow=False, color=color)
            return id
        elif grasp_tf.ndim == 3:
            ids = []
            for i in range(len(grasp_tf)):
                if round(score[i], 4) == 0:
                    continue
                grasp_point = np.array([0, 0, 0])
                grasp_dir = np.array([0, 0, length_coef*3*score[i]])
                points_transformed = trimesh.transform_points(
                    [grasp_point, grasp_dir], grasp_tf[i])
                grasp_point = np.array(points_transformed[0])
                grasp_dir = np.array(points_transformed[1])
                id = self.plot_vector(grasp_point, grasp_dir,
                                 radius_cyl=length_coef/10, arrow=False, color=color)
                ids.append(id)
            return ids
            

    def plot_coordinate_system(self, tf=None, scale=1, id = None):
        """
        Plot a coordinate system given a transformation matrix and a scale
        --------------
        Arguments:
            tf : np.array(4,4)
                Transformation matrix
            scale : float
                Scale of the coordinate system
        Return:
        --------------
            None : None
        """
        if type(tf) == None:
            tf = np.eye(4)
        if id is None:
            id = id_generator("cs_")
        # Create coordinate system
        cs = trimesh.creation.axis(origin_size=scale, transform=tf)
        # Append to meshes
        self.meshes[id] = cs

        return id

    def remove_feature(self, id):
        """
        Removes a feature from the viewer
        --------------
        Arguments:
            - id : string
                Id of the feature to be removed
        Return:
        --------------
            - success : bool
                True if feature was removed, False otherwise
        """

        if id in self.meshes:
            del self.meshes[id]
            return True
        else:
            return False
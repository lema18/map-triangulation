Implementación de un sistema de localización monocular.
Para poder utilizarlo se requieren las siguientes dependencias:
-> La librería OpenCV en la versión 3.0 o superior: https://opencv.org/releases/
-> La librería g2o: https://github.com/RainerKuemmerle/g2o
-> La librería PCL: http://pointclouds.org/
-> La librería Eigen para álgebra lineal: http://eigen.tuxfamily.org/index.php?title=Main_Page
-> La librería DboW2 para la creación de vocabularios a través del modelo bag of words: https://github.com/dorian3d/DBoW2

Una vez instaladas todas las dependencias y compilada la aplicación se le deben proporcionar los siguientes parámetros de entrada:
argv[1]= Ruta a la carpeta en la que tenga almacenadas las imágenes del dataset. Cada imagen debe tener un nombre del tipo
         left_i, donde i es el número de la imagen.
argv[2]= Número de imágenes utilizadas para la creación del mapa inicial. Se recomienda que sea 50.
argv[3]= Número de imágenes del dataset. 
argv[4]= Ruta a la carpeta que tenga almacenada el vocabulario de palabras visuales de la imagen. Se proporciona dicho
         vocabulario para el dataset freiburg1_xyz de TUM dataset. Este vocabulario está en el archivo small_voc.yml.gz
argv[5]= variable para activar el módulo de optimización local(1 para activar, 0 para desactivar)

El programa ha sido probado para una secuencia de 300 imágenes del dataset freiburg1_xyz de TUM: https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_xyz
Se proporciona la trayectoría real para compararla con la trayectoría que proporciona nuestro programa.
La trayectoria real se almacena en el archivo "freiburg_xyz_groundtruth.txt" que se proporciona en este repositorio.
La trayectoria obtenida por nuestro programa se almacena el el archivo "odometry.txt" que se genera tras ejecutar la aplicación.

Para comparar las trayectorias se propone el archivo vis.py, que se encarga de generar una representación gráfica 
de ambas trayectorias. El script recibe los siguientes parámetros de entrada:
argv[1]= Ruta a la carpeta en la que se almacena el archivo "freiburg_xyz_groundtruth.txt".
argv[2]= Ruta a la carpeta en la que se almacena el archivo "odometry.txt".

Si el usuario desea probar otro dataset, tendría que modificar los parámetros intrínsecos de la cámara en el archivo "main.cpp".
Además, tambien tendría que generar un vocabulario visual para su dataset con la ayuda de la librería DboW2.

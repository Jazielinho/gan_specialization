import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def mostrar_imagenes(image_array, num_images=25, size=(28, 28)):
    '''
    Funcion para mostrar imagenes
    '''
    image_plot = image_array.reshape(-1, *size) * 255
    image_plot = image_plot.astype(np.uint8)
    for i in range(num_images):
        plt.subplot(5, int(num_images) / 5, i + 1)
        plt.imshow(image_plot[i].astype(np.uint8))
        plt.axis('off')
    plt.show()


def crear_generador(dim_z: int = 10, dim_imagen: int = 784, dim_oculta: int = 128):
    ''' Generador de imágenes '''

    def obtener_bloque_generador(dim_entrada: int, dim_salida: int) -> tf.keras.Model:
        ''' Genera una capa lineal con normalización por bloques '''
        entrada = tf.keras.Input(shape=dim_entrada)
        capa_lineal = tf.keras.layers.Dense(units=dim_salida)(entrada)
        normalizacion = tf.keras.layers.BatchNormalization()(capa_lineal)
        salida_relu = tf.keras.activations.relu(normalizacion)
        return tf.keras.Model(inputs=entrada, outputs=salida_relu)

    entrada = tf.keras.Input(shape=dim_z)
    salida_1 = obtener_bloque_generador(dim_entrada=dim_z, dim_salida=dim_oculta)(entrada)
    salida_2 = obtener_bloque_generador(dim_entrada=dim_oculta, dim_salida=dim_oculta * 2)(salida_1)
    salida_3 = obtener_bloque_generador(dim_entrada=dim_oculta * 2, dim_salida=dim_oculta * 4)(salida_2)
    salida_4 = obtener_bloque_generador(dim_entrada=dim_oculta * 4, dim_salida=dim_oculta * 8)(salida_3)
    salida_lineal = tf.keras.layers.Dense(units=dim_imagen)(salida_4)
    salida_sigmoide = tf.keras.activations.sigmoid(salida_lineal)

    return tf.keras.Model(inputs=entrada, outputs=salida_sigmoide)


def obtener_ruido(numero_muestra: int, dim_z: int) -> tf.random.normal:
    ''' Genera vector aleatorio de dimension (numero_muestra, dim_z) '''
    return tf.random.normal((numero_muestra, dim_z))


def crear_discriminador(dim_imagen: int = 784, dim_oculta: int = 128) -> tf.keras.Model:
    ''' Discriminador de imágenes '''

    def obtener_bloque_discriminador(dim_entrada: int, dim_salida: int) -> tf.keras.Model:
        ''' Genera una capa lineal con normalización por bloques '''
        entrada = tf.keras.Input(shape=dim_entrada)
        capa_lineal = tf.keras.layers.Dense(units=dim_salida)(entrada)
        salida_leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)(capa_lineal)
        return tf.keras.Model(inputs=entrada, outputs=salida_leaky_relu)

    entrada = tf.keras.Input(shape=dim_imagen)
    salida_1 = obtener_bloque_discriminador(dim_entrada=dim_imagen, dim_salida=dim_oculta*4)(entrada)
    salida_2 = obtener_bloque_discriminador(dim_entrada=dim_oculta*4, dim_salida=dim_oculta*2)(salida_1)
    salida_3 = obtener_bloque_discriminador(dim_entrada=dim_oculta*2, dim_salida=dim_oculta)(salida_2)
    salida_lineal = tf.keras.layers.Dense(units=1)(salida_3)
    return tf.keras.Model(inputs=entrada, outputs=salida_lineal)


def obtener_perdida_discriminador(salida_real, salida_falsa):
    perdida_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(salida_real), salida_real)
    perdida_falsa = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(salida_falsa), salida_falsa)
    perdida_total = perdida_real + perdida_falsa
    return perdida_total


def obtener_perdida_generador(salida_falsa):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(salida_falsa), salida_falsa)


def obtener_mnist_data(batch_size=128):
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = tf.keras.datasets.mnist.load_data()
    mnist_x_train = mnist_x_train.reshape(-1, 784)
    mnist_x_test = mnist_x_test.reshape(-1, 784)
    mnist_data = np.concatenate((mnist_x_train, mnist_x_test), axis=0)
    mnist_data = mnist_data / 255.0
    mnist_target = np.concatenate((mnist_y_train, mnist_y_test), axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((mnist_data, mnist_target))
    dataset = dataset.shuffle(len(mnist_data)).batch(batch_size)
    return dataset


def entrenamiento_GAN(dim_z, epochs, batch_size):

    dataset = obtener_mnist_data(batch_size=batch_size)

    generador = crear_generador(dim_z=dim_z, dim_imagen=784, dim_oculta=128)
    discriminador = crear_discriminador(dim_imagen=784, dim_oculta=128)

    generador_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminador_optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in tqdm.tqdm(range(epochs)):
        generador_perdida = 0
        discriminador_perdida = 0

        for imagenes_batch, _ in dataset:
            numero_muestra = len(imagenes_batch)

            ruido = obtener_ruido(numero_muestra=numero_muestra, dim_z=dim_z)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                imagenes_generadas = generador(ruido, training=True)

                salida_real = discriminador(imagenes_batch, training=True)
                salida_falsa = discriminador(imagenes_generadas, training=True)

                perdida_discriminador = obtener_perdida_discriminador(salida_real=salida_real, salida_falsa=salida_falsa)
                perdida_generador = obtener_perdida_generador(salida_falsa=salida_falsa)

            gradientes_discriminador = disc_tape.gradient(perdida_discriminador, discriminador.trainable_weights)
            gradientes_generador = gen_tape.gradient(perdida_generador, generador.trainable_weights)

            discriminador_optimizer.apply_gradients(zip(gradientes_discriminador, discriminador.trainable_weights))
            generador_optimizer.apply_gradients(zip(gradientes_generador, generador.trainable_weights))

            generador_perdida += perdida_generador
            discriminador_perdida += perdida_discriminador

        print(f"Epoch: {epoch}, perdida generador: {generador_perdida}, perdida discriminador: {discriminador_perdida}")

    return generador, discriminador


def main():
    generador, discriminador = entrenamiento_GAN(dim_z=64, epochs=20, batch_size=1024)


if __name__ == '__main__':
    main()
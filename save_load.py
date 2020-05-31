import tensorflow as tf
import os

class LoadSave():

    @staticmethod
    def files_exist(load_dir):
        subdirs = [s for s in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, s))]
        subdirs.sort()
        if not subdirs: return []
        subdir = subdirs[-1]

        discA_path = os.path.join(load_dir, subdir, 'model_discA')
        discB_path = os.path.join(load_dir, subdir, 'model_discB')
        genA2B_path = os.path.join(load_dir, subdir, 'model_genA2B')
        genB2A_path = os.path.join(load_dir, subdir, 'model_genB2A')

        if os.path.isdir(discA_path) and os.path.isdir(discB_path) and \
            os.path.isdir(genA2B_path) and os.path.isdir(genB2A_path):
            return discA_path, discB_path, genA2B_path, genB2A_path, subdir
        return []


    @staticmethod
    def load_files(strategy, load, load_dir):
        if load == True:
            files = LoadSave.files_exist(load_dir)
            if files:
                with strategy.scope():
                    discA = tf.keras.models.load_model(files[0])
                    discB = tf.keras.models.load_model(files[1])
                    genA2B = tf.keras.models.load_model(files[2])
                    genB2A = tf.keras.models.load_model(files[3])

                    print('Models loaded from %s' % files[-1])
                    return discA, discB, genA2B, genB2A
            else:
                return None
        return None


    @staticmethod
    def save_models(save_dir, discA, discB, genA2B, genB2A, epoch, batch):
        subdir = 'E' + str(epoch) + 'B' + str(batch)
        discA.save(os.path.join(save_dir, subdir, 'model_discA'), save_format='tf')
        discB.save(os.path.join(save_dir, subdir, 'model_discB'), save_format='tf')
        genA2B.save(os.path.join(save_dir, subdir, 'model_genA2B'), save_format='tf')
        genB2A.save(os.path.join(save_dir, subdir, 'model_genB2A'), save_format='tf')
        print('Models saved')
        return
"""

"""
import os
import glob
import depthwise_segnet
import segnet
import helper
from parameters import train_data_folder, vali_data_folder, test_data_folder
from parameters import image_shape, input_shape, num_classes, class_colors
import data_processing


if __name__ == "__main__":
    DEBUG = False

    learning_rate = 2e-4
    epochs = 150
    batch_size = 16

    # Download the data if necessary

    data_processing.maybe_download_data()
    data_processing.maybe_create_test_data()

    train_images = sorted(glob.glob(os.path.join(train_data_folder, 'images', '*.jpeg')))
    train_masks = sorted(glob.glob(os.path.join(train_data_folder, 'masks', '*.png')))
    vali_images = sorted(glob.glob(os.path.join(vali_data_folder, 'images', '*.jpeg')))
    vali_masks = sorted(glob.glob(os.path.join(vali_data_folder, 'masks', '*.png')))
    test_images = sorted(glob.glob(os.path.join(test_data_folder, 'images', '*.jpeg')))
    test_masks = sorted(glob.glob(os.path.join(test_data_folder, 'masks', '*.png')))

    num_train_data = len(train_images)
    num_vali_data = len(vali_images)
    num_test_data = len(test_images)

    assert (num_train_data == len(train_masks) == 4131)
    assert (num_vali_data == len(vali_masks) == 584)
    assert (num_test_data == len(test_masks) == 600)

    # helper.check_environment()

    if DEBUG is True:
        epochs = 2
        batch_size = 2
        num_vali_data = 2
        num_train_data = 2
        num_test_data = 2

    #######################################################################
    ### The original segnet (without pooling index)
    #######################################################################
    weights_file = "segnet_weights.h5"
    structure_file = "segnet_model.txt"
    output_folder = "./segnet_inference"
    loss_history_file = "segnet_loss.pkl"

    model = segnet.build_model(image_shape, input_shape, num_classes)
    helper.show_model(model, structure_file)

    helper.train(model, epochs, batch_size, learning_rate,
                 class_colors, train_data_folder, num_train_data,
                 vali_data_folder, num_vali_data,
                 weights_file, loss_history_file)

    helper.output_prediction(model, image_shape, class_colors, batch_size,
                             test_data_folder, num_test_data, output_folder)

    #######################################################################
    ### The depthwise segnet
    ######################################################################
    weights_file = "depthwise_segnet_weights.h5"
    structure_file = "depthwise_segnet_model.txt"
    output_folder = "./depthwise_segnet_inference"
    loss_history_file = "depthwise_segnet_loss.pkl"

    model = depthwise_segnet.build_model(image_shape, input_shape, num_classes)
    helper.show_model(model, structure_file)

    helper.train(model, epochs, batch_size, learning_rate,
                 class_colors, train_data_folder, num_train_data,
                 vali_data_folder, num_vali_data,
                 weights_file, loss_history_file)

    helper.output_prediction(model, image_shape, class_colors, batch_size,
                             test_data_folder, num_test_data, output_folder)

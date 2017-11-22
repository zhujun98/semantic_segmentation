"""

"""
import data_processing
import depthwise_segnet
import segnet
import helper
from parameters import train_data_folder, vali_data_folder, test_data_folder
from parameters import num_train_data, num_vali_data, num_test_data
from parameters import image_shape, input_shape, num_classes, class_colors


if __name__ == "__main__":
    learning_rate = 2e-4
    batch_size = 8
    epochs = 2

    helper.check_environment()

    data_processing.maybe_download_data()

    #######################################################################
    ### The original segnet (without pooling index)
    #######################################################################

    model = segnet.build_model(image_shape, input_shape, num_classes)
    helper.show_model(model, "segnet_model.txt")

    weights_file = "saved_segnet.h5"
    helper.train(model, epochs, batch_size, learning_rate, class_colors,
                 train_data_folder, batch_size, vali_data_folder, batch_size,
                 weights_file)

    helper.output_prediction(model, image_shape, class_colors, batch_size,
                             test_data_folder, "./output_segnet")

    #######################################################################
    ### The depthwise segnet
    #######################################################################

    model = depthwise_segnet.build_model(image_shape, input_shape, num_classes)
    helper.show_model(model, "depthwise_segnet_model.txt")

    weights_file = "saved_depthwise_segnet.h5"
    helper.train(model, epochs, batch_size, learning_rate, class_colors,
                 train_data_folder, batch_size, vali_data_folder, batch_size,
                 weights_file)

    helper.output_prediction(model, image_shape, class_colors, batch_size,
                             test_data_folder, "./output_depthwise_segnet")
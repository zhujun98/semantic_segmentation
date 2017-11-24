"""

"""
import depthwise_segnet
import segnet
import helper
from parameters import train_data_folder, vali_data_folder, test_data_folder
from parameters import num_train_data, num_vali_data, num_test_data
from parameters import image_shape, input_shape, num_classes, class_colors


if __name__ == "__main__":
    DEBUG = False

    learning_rate = 2e-4
    epochs = 150
    batch_size = 16
    max_preds = 1e4
    if DEBUG is True:
        epochs = 2
        batch_size = 2
        num_vali_data = 2
        num_train_data = 2
        max_preds = 2

    helper.check_environment()

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
                             test_data_folder, output_folder,
                             max_predictions=max_preds)

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
                             test_data_folder, output_folder,
                             max_predictions=max_preds)

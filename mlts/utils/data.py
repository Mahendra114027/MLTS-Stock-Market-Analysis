def split_data(data):
    """
    Split data into training and testing sets
    """
    train_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    return train_data, test_data

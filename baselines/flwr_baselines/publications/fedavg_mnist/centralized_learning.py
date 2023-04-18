from flwr_baselines.publications.fedavg_mnist import model
from flwr_baselines.publications.fedavg_mnist.dataset import load_datasets
import torch

def run_CL(cfg):

    DEVICE = torch.device("cuda:0")
    # DEVICE = torch.device("cpu")
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("*Centralized training*")

    print("Load data")
    trainloaders, valloaders, testloader = load_datasets(
        #iid=iid, balance=balance, num_clients=num_clients, batch_size=batch_size
        iid=cfg.iid, balance=cfg.balance, num_clients=1, batch_size=cfg.batch_size # declare num_clients to 1
    )

    trainloader = trainloaders[0]
    valloader = valloaders[0]
    net = model.Net().to(DEVICE)

    # num_epochs = cfg.num_epochs
    num_epochs = cfg.num_rounds # Using this instead to test


    print("Start training")
    training_loss = [None]*(num_epochs+1)
    training_accuracy = [None]*(num_epochs+1)
    training_loss[0], training_accuracy[0] = model.test(net, trainloader, device = DEVICE)
    #print(f"Epoch {0}: training loss {training_loss[0]:.4f}, training accuracy {validation_accuracy[0]:.4f}")
    validation_loss = [None]*(num_epochs+1)
    validation_accuracy = [None]*(num_epochs+1)
    validation_loss[0], validation_accuracy[0] = model.test(net, valloader, device = DEVICE)
    print(f"Epoch {0}: training loss {training_loss[0]:.4f}, training accuracy {training_accuracy[0]:.4f}, validation loss {validation_loss[0]:.4f}, validation accuracy {validation_accuracy[0]:.4f}")
    
    for epoch in range(num_epochs):
        model.train(net, trainloader, epochs=1, device = DEVICE, learning_rate = cfg.learning_rate)
        # MOVE ABOVE LINE INSTEAD OF LOOP (on change to num_epochs)
        training_loss[epoch+1], training_accuracy[epoch+1] = model.test(net, trainloader, device = DEVICE)
        validation_loss[epoch+1], validation_accuracy[epoch+1] = model.test(net, valloader, device = DEVICE)
        print(f"Epoch {epoch+1}: training loss {training_loss[epoch+1]:.4f}, training accuracy {training_accuracy[epoch+1]:.4f}, validation loss {validation_loss[epoch+1]:.4f}, validation accuracy {validation_accuracy[epoch+1]:.4f}")

    print("Evaluate model")
    test_loss, test_accuracy = model.test(net, testloader, device = DEVICE)
    print(f"Final test set performance:\n\tloss {test_loss:.4f}\n\taccuracy {test_accuracy:.4f}")

    print("*END: Centralized training*")

    return training_loss, training_accuracy, validation_loss, validation_accuracy, test_loss, test_accuracy


















##################### 2023-03-06 14:26 (Below)

# from flwr_baselines.publications.fedavg_mnist import model
# from flwr_baselines.publications.fedavg_mnist.dataset import load_datasets
# import torch

# def run_CL(cfg):

#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("*Centralized training*")

#     print("Load data")
#     trainloaders, valloaders, testloader = load_datasets(
#         #iid=iid, balance=balance, num_clients=num_clients, batch_size=batch_size
#         iid=cfg.iid, balance=cfg.balance, num_clients=1, batch_size=cfg.batch_size # declare num_clients to 1
#     )

#     trainloader = trainloaders[0]
#     valloader = valloaders[0]
#     net = model.Net().to(DEVICE)

#     print("Start training")
#     # validation_loss = []
#     # validation_accuracy = []
#     validation_loss = [None]*(cfg.num_epochs+1)
#     validation_accuracy = [None]*(cfg.num_epochs+1)
#     validation_loss[0], validation_accuracy[0] = model.test(net, valloader, device = DEVICE)
#     print(f"Epoch {0}: validation loss {validation_loss[0]:.4f}, accuracy {validation_accuracy[0]:.4f}")
    
#     for epoch in range(cfg.num_epochs):
#         model.train(net, trainloader, epochs=1, device = DEVICE, learning_rate = cfg.learning_rate)
#         # MOVE ABOVE LINE INSTEAD OF LOOP (on change to cfg.num_epochs)
#         # #loss, accuracy = model.test(net, valloader, device = DEVICE)
#         # loss, accuracy = model.test(net, valloader, device = DEVICE)
#         # validation_loss.append(loss)
#         # validation_accuracy.append(accuracy)
#         validation_loss[epoch+1], validation_accuracy[epoch+1] = model.test(net, valloader, device = DEVICE)
#         print(f"Epoch {epoch+1}: validation loss {validation_loss[epoch+1]:.4f}, accuracy {validation_accuracy[epoch+1]:.4f}")

#     # # for epoch in range(cfg.num_epochs):
#     # model.train(net, trainloader, epochs=cfg.num_epochs, device = DEVICE, learning_rate = cfg.learning_rate)
#     # # MOVE ABOVE LINE INSTEAD OF LOOP (on change to cfg.num_epochs)
#     # loss, accuracy = model.test(net, valloader, device = DEVICE)
#     # #print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

#     print("Evaluate model")
#     test_loss, test_accuracy = model.test(net, testloader, device = DEVICE)
#     print(f"Final test set performance:\n\tloss {test_loss:.4f}\n\taccuracy {test_accuracy:.4f}")

#     print("*END: Centralized training*")

#     return validation_loss, validation_accuracy, test_loss, test_accuracy
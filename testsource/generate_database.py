import generate_dataset.generate_dataset_csv
import my_dataset

generate_dataset_csv./faces_database('./')

input_file_path = './faces_database.csv'
ROOT_DIR = ""
imgDatabase = MyDataset(input_file_path, ROOT_DIR, transform=transforms.Compose([
transforms.ToTensor()]))
database_loader = torch.utils.data.DataLoader(imgDatabase, batchsize = 100, shuffle=False)

from modelVQA import EasyQADataset
test_dataset = EasyQADataset(
                           df=test_df,
                           image_encoder = image_encoder,
                           text_encoder = text_encoder,
                           tokenizer = tokenizer,
                           image_processor = image_processor
                           )

device = "cuda:0"
model.load_state_dict(torch.load('/DATA/pal14/M22MA002/EasyVqa/models/easyvqa_finetuned_epoch_8.model'))
model.to(device)

dataloader_test = DataLoader(test_dataset,
                            sampler=SequentialSampler(test_dataset),
                            batch_size=128)

_, preds, truths, confidence = evaluate(dataloader_test)

"""# <font color="red"> 10. Moment of Truth :-)? </font>"""

print("Test Acc with Resnet50d: " , accuracy_score_func(preds,truths))

print("Test Acc with ViT: " , accuracy_score_func(preds,truths))

test_results_df = pd.concat([test_df, pd.DataFrame(preds, columns=["preds"]), pd.DataFrame(truths, columns=["gt"]), pd.DataFrame(confidence, columns=["confidence"])], axis=1)

test_results_df.sample(5)

"""# <font color="red"> Is the Model Overconfident when its wrong? </font>"""

test_results_df[(test_results_df["preds"] != test_results_df["gt"]) & (test_results_df["confidence"] >= 0.90)].shape

label2idx

from PIL import Image
Image.open("/usr/local/lib/python3.7/dist-packages/easy_vqa/data/test/images/130.png")
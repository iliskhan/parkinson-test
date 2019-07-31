from keras.callbacks import ModelCheckpoint

from scripts.model import load_data, build_model, compile_model, len_window, model_file_name

train_x, train_y, test_x, test_y = load_data()
model = compile_model(build_model())

model_checkpoint = ModelCheckpoint(filepath=f'../models/parkinson_window={len_window}.tmp.h5', save_best_only=True)
model.fit(x=train_x, y=train_y, epochs=20, callbacks=[model_checkpoint])

model.save(model_file_name)

preds = model.evaluate(x=test_x, y=test_y)

print('Test loss =', preds[0])
print('Test accuracy =', preds[1])

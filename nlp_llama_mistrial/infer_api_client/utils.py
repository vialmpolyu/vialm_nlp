from sentence_transformers import SentenceTransformer, util

def get_task_id(input_text,query_embeddings, text_model):
    input_embedding = text_model.encode(input_text)
    similarities = util.cos_sim(input_embedding, query_embeddings)
    most_similar_query_index = similarities.argmax()
    return most_similar_query_index

def get_obj_list_from_cv(cv_results):
    # Processing frame_data
    cls_dict = cv_results[0].names

    cls = cv_results[0].boxes.cls.cpu().numpy()
    cls_dict_img = {}
    for idx, c in enumerate(cls):
        cls_dict_img[idx] = cls_dict[int(c)]

    boxxyxy = cv_results[0].boxes.xyxy.cpu().numpy()
    obj_list = []
    for idx, elm in enumerate(boxxyxy):
        obj = {}
        x1, y1, x2, y2 = elm
        coord = ((int(x1), int(y1)), (int(x2), int(y2)))
        obj['idx'] = idx
        obj['object'] = cls_dict_img[idx]
        obj['coordinate'] = coord
        obj_list.append(obj)
    return obj_list


def ocr_detect(ocr_out):
    out=ocr_out
    text = ''
    for i in range(len(out)):
        if out[i]['score'] > 0.5:
            coords = out[i]['position']
            minX = 100000
            maxX = -1
            minY = 100000
            maxY = -1
            for coord in coords:
                x = int(coord[0])
                y = int(coord[1])
                minX = minX if minX < x else x
                minY = minY if minY < y else y
                maxX = maxX if maxX > x else x
                maxY = maxY if maxY > y else y
            if len(out[i]['text']) <= 0:
                print("skip")
                continue
            text = text + out[i]['text'] + '\n'
    return text
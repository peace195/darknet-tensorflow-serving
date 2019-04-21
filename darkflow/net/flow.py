import time
import pickle
from multiprocessing.pool import ThreadPool
import os
import numpy as np
import math
import glob
import json
import os
import shutil
import sys
import math
import xml.etree.ElementTree as ET

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss
    best_mAP = -1
    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]

        if not ckpt:
            predict(self)
            mAP = evaluate(self, self.FLAGS.imgdir, os.path.join(self.FLAGS.imgdir, 'out'), "results_" + str(ckpt) + ".txt")
            if mAP > best_mAP:
                best_mAP = mAP
                _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo


def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))


def evaluate(self, test_dir, ret_dir, results_file_name):
    MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
    GT_PATH = os.path.join(os.getcwd(), test_dir)
    DR_PATH = os.path.join(os.getcwd(), ret_dir)

    xml_list = glob.glob(GT_PATH + '/*.xml')
    if len(xml_list) == 0:
        print("Error: no .xml files found in ground-truth")
        sys.exit()
    for tmp_file in xml_list:
        with open(tmp_file.replace(".xml", ".txt"), "w") as new_f:
            root = ET.parse(tmp_file).getroot()
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                obj_name = "_".join(obj_name.split())
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    
    json_list = glob.glob(DR_PATH + '/*.json')
    if len(json_list) == 0:
        print("Error: no .json files found in detection-results")
        sys.exit()
    for tmp_file in json_list:
        with open(tmp_file.replace(".json", ".txt"), "w") as new_f:
            data = json.load(open(tmp_file))
            for obj in data:
                obj_name = "_".join(obj['label'].split())
                conf = obj['confidence']
                left = obj['topleft']['x']
                top = obj['topleft']['y']
                right = obj['bottomright']['x']
                bottom = obj['bottomright']['y']
                new_f.write(
                    obj_name + " " + str(conf) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(
                        bottom) + '\n')


    def log_average_miss_rate(precision, fp_cumsum, num_images):
        if precision.size == 0:
            lamr = 0
            mr = 1
            fppi = 0
            return lamr, mr, fppi

        fppi = fp_cumsum / float(num_images)
        mr = (1 - precision)

        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)

        # Use 9 evenly spaced reference points in log-space
        ref = np.logspace(-2.0, 0.0, num = 9)
        for i, ref_i in enumerate(ref):
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]

        # log(0) is undefined, so we use the np.maximum(1e-10, ref)
        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

        return lamr, mr, fppi

    def error(msg):
        print(msg)
        sys.exit(0)


    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """
    def voc_ap(rec, prec):
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre


    def file_lines_to_list(path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content

    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
    results_files_path = "results"
    if not os.path.exists(results_files_path):
        os.makedirs(results_files_path)

    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                else:
                        class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip

            bbox = left + " " + top + " " + right + " " +bottom
            if is_difficult:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                    is_difficult = False
            else:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                    # count that object
                    if class_name in gt_counter_per_class:
                        gt_counter_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        gt_counter_per_class[class_name] = 1

                    if class_name not in already_seen_classes:
                        if class_name in counter_images_per_class:
                            counter_images_per_class[class_name] += 1
                        else:
                            # if class didn't exist yet
                            counter_images_per_class[class_name] = 1
                        already_seen_classes.append(class_name)


        # dump bounding_boxes into a ".json" file
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            #print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}

    results_file = open(results_files_path + "/" + results_file_name, 'w')
    results_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # set minimum overlap
            min_overlap = MINOVERLAP

            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                    else:
                        fp[idx] = 1
            else:
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        print(text)
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
        lamr_dictionary[class_name] = lamr


    results_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    results_file.write(text + "\n")
    print(text)

    shutil.rmtree(TEMP_FILES_PATH)

    # iterate through all the files
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        # get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # count that object
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    results_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    results_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        results_file.write(text)
    results_file.close()
    return mAP
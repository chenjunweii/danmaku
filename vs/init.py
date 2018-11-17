import mxnet as mx
from evaluate import evaluation

class init(object):

    def __init__(self):

        pass

    def Init(self, ):

        ewt = dict()

        fscore = dict()

        precision = dict()
        
        recall = dict()
        
        fscore = []
        
        precision = []
        
        recall = []

        steps = []

        if self.checkpoint is not None:

            print ('[*] Restore From CheckPoint => {}'.format(self.checkpoint))

            self.net.load_params(self.checkpoint, ctx = device)
                
            s = int(self.checkpoint.split('_')[-1].split('.')[0])
            
            savelogpath = os.path.join("log", arch, '{}_{}.h5'.format(prefix, s))
            
            h5 = h5py.File(savelogpath, 'r')

            steps = list(np.asarray(h5['step']))

            fscore = list(np.asarray(h5['f-score']))
            
            lr = float(np.asarray(h5['lr']).astype(float))

        lr_scheduler = mx.lr_scheduler.FactorScheduler(self.lr_decay_step, self.lr_decay_rate)

        if self.arch == 'gan':

            G_optimizer = 'adam'
                
            G_options = {'learning_rate': lr,
                       'lr_scheduler' : lr_scheduler,
                       'clip_gradient': 0.5,
                       #'momentum' : 0.9,
                       'wd' : 0.001}
            
            D_optimizer = 'adam'
            
            D_options = {'learning_rate': lr,
                       'lr_scheduler' : lr_scheduler,
                       'clip_gradient': 0.5,
                       #'momentum' : 0.9,
                       'wd' : 0.001}

            self.set_optimizer(G_optimizer, G_options, D_optimizer, D_options)
        
        else:

            optimizer = 'adam'
        
            trainer = mx.gluon.Trainer(net.collect_params(), optimizer, options)

        print ('[*] Start Training ...')

        print('[*] Evaluation Metric : {}'.format(self.metric.title()))

        self.evaluater = evaluate_while_train(net, nds, arch, self.d, self.datatype, self.metric, device, self.shot)

        first = True

        i = 0
        
        means = []
        
        mins = []
        
        maxs = []

        try:

            print('[*] Target Iteration : ', n_iters)

            print('[*] Start From Iteration : ', s)

            while s < n_iters and data.epoch < self.epoch:

                if s % es == 0 and s != 0:

                    first = False

                    fscore['step'].append(s)

                    #fscore['summe'].append(ewt['summe'].evaluate())

                    #fscore[self.d].append(ewt[self.d].evaluate(net, nds) * 100)
                    
                    f, p, r = ewt[self.d].evaluate(net, nds)

                    fscore[self.d].append(np.mean(f) * 100)
                    
                    precision[self.d].append(np.mean(p) * 100)
                    
                    recall[self.d].append(np.mean(r) * 100)

                    precision['step'] = fscore['step']
                    
                    recall['step'] = fscore['step']

                    print('--------------------------------------')

                    print('[*] {} Evaluation F-Score: {:.2f}'.format(self.d, fscore[self.d][-1]))
                    
                    print('[*] {} Evaluation Precision : {:.2f}'.format(self.d, precision[self.d][-1]))
                    
                    print('[*] {} Evaluation Recall : {:.2f}'.format(self.d, recall[self.d][-1]))

                    plot(fscore, arch, self.d, prefix, s, 'f-score')
                    
                    plot(precision, arch, self.d, prefix, s, 'precision')
                    
                    plot(recall, arch, self.d, prefix, s, 'recall')

                    plot_table(f, p, r, arch, prefix)

                    print('[*] {} F : {}'.format(self.d, (2 * precision[self.d][-1] * recall[self.d][-1]) / (precision[self.d][-1] + recall[self.d][-1])))
                    
                    savepath = os.path.join('{}_pretrained.mx'.format(self.d))
                    
                    print('[!] Checkpoint is save to {}'.format(savepath))
                    
                    print('--------------------------------------')
                    
                    net.save_parameters(savepath)

                # batch

                seqlist, tarlist, bdlist = data.next()
                
                
                if self.datatype == 'raw':

                    seqlist, tarlist = preprocess_list(seqlist, tarlist, bd_list = bdlist)

                current_batch = len(seqlist)

                olen = [seq.shape[0] for seq in seqlist] # original length

                nps['input'] = pad(seqlist, olen).swapaxes(0, 1)

                nds['input'] = nd.array(nps['input'], device)

                #nds['target'] = nd.array(tarlist.swapaxes(0,1), device)

                nps['target'] = pad(tarlist, olen).swapaxes(0, 1)

                nds['target'] = nd.array(nps['target'], device)

               # print('target', nps['target'][0 : 100, 0])
                
                if arch == 'gan':
                    
                    prediction = net(nds['input'], nds['target'], 0.15, 'train')

                else:

                    if arch == 'dpp':

                        nps['shot'] = shotlist[0]

                        nds['shot'] = nd.array(nps['shot'], device)

                    with autograd.record():

                        if arch == 'edwa':

                            prediction = net(nds['input'], [nds['encoder_state_h'], nds['encoder_state_c']])

                        elif arch == 'ewd' or arch == 'mpewd':

                            prediction = net(nds['input'], [nds['encoder_state_h'], nds['encoder_state_c']])

                        elif arch == 'dpp':
                            prediction, dpp_loss = net(nds['input'], nds['shot'])
                            assert(minibatch == 1)

                        elif arch == 'gan':
                            prediction = net(nds['input'], nds['target'], 0.15, 'train')
                        else:
                            prediction = net(nds['input'])

                        pad_mask, active = loss_mask(current_batch, olen)
                       
                        loss = nd.sum(cross_entropy_2(prediction, nds['target']) * nd.array(pad_mask, device)) / float(np.sum(pad_mask))

                if s % 10 == 0:

                    if arch == 'gan':

                        print('-' * 20)
                        print('[*] Step {} G Loss {} LR {}'.format(s, prediction[0].mean().asnumpy(), net.trainerG.learning_rate))
                        print('[*] Step {} D Loss {} LR {}'.format(s, prediction[1].mean().asnumpy(), net.trainerG.learning_rate))
                    
                    else:

                        print('[*] Step {} Loss {} LR {}'.format(s, loss.asnumpy(), trainer.learning_rate))

                if s % self.ss == 0 and s != 0:
                    
                    try:

                        savelogpath = os.path.join("log", arch, '{}_{}.h5'.format(prefix, s))
                        
                        with h5py.File(savelogpath) as h5:

                            h5['f-score'] = np.asarray(fscore['step'])

                            h5['step'] = np.asarray(fscore[self.d])

                            h5['lr'] = np.asarray(trainer.learning_rate)

                        print('[!] Log is save to {}'.format(savelogpath))
                    
                    except:

                        print('[!] Log is Not save to {}'.format(savelogpath))

                    savepath = os.path.join("..", "mx", arch, '{}_{}.mx'.format(prefix, s))
                   
                    print('[!] Checkpoint is save to {}'.format(savepath))
                    
                    net.save_params(savepath)

                if arch != 'gan':
                    
                    loss.backward()
                
                    trainer.step(minibatch)

                s += 1

            savepath = os.path.join("..", "mx", arch, '{}_{}.mx'.format(prefix, s))
            
            savelogpath = os.path.join("log", arch, '{}_{}.h5'.format(prefix, s))
            
            h5 = h5py.File(savelogpath, 'w')

            h5['f-score'] = np.asarray(fscore['step'])

            h5['step'] = np.asarray(fscore[self.d])

            h5['lr'] = np.asarray(trainer.learning_rate)

            h5.close()
            
            print('[!] Log is save to {}'.format(savelogpath))

            print('[!] Checkpoint is save to {}'.format(savepath))
            
            net.save_params(savepath)
        
        except KeyboardInterrupt:
        
            print ('Training interrupted.')

        

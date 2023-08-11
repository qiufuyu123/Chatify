import asyncio, os
from typing import Any, Coroutine, List, Optional, Union,Dict
from wechaty_puppet import EventHeartbeatPayload, EventReadyPayload, MessageType
from wechaty_puppet import FileBox  # type: ignore
import time
from bert_dataset import cute_labels
from wechaty import Wechaty, Contact
from wechaty.user import Friendship, Message, Room
import baidu
import psutil
import torch
from bert_mbti import MBTIMain
from chatgpt import gpt_35_api_stream
from threading import Thread,Lock
predictor = MBTIMain()
print('==========   BERT * 4   ==========')
print('start loading models...')
predictor.loadmodule()
print('Bot is starting now!')

class UserNode:
    def __init__(self):
        self.ctx = []
        self.trans = ''
        self.state = 0
        self.timestamp=0
        self.pred =''
        self.recom = ''
        self.require = ''
        self.mbti=''
    def set_state(self,s):
        if s == 1:
            self.set_time()
        self.state = s
        
    def get_state(self):
        return self.state
    def set_mbti(self,mbti):
        self.mbti = mbti
    def get_mbti(self):
        return self.mbti
    def set_output(self,text):
        self.recom = text
    def process(self):
        self.sendtranslate()
        if self.trans=='':
            return False,'error'
        total,pct1,pct2,pct3,pct4 = self.predict()
        return True,total
    def set_time(self):
        self.timestamp = time.time()
    
    def istimeout(self,timeout):
        return time.time() - self.timestamp > timeout    
    
    def add_ctx(self,text):
        self.ctx.append(text)
    
    def sendtranslate(self):    
        full =''
        for s in self.ctx:
            full+=s+','
        self.trans = baidu.trans(full)
        print(self.trans)

    def predict(self):
        return predictor.predict(self.trans)
    def get_output(self):
        return self.recom
    def recommend(self,type):
        last = ''
        if len(self.ctx)<3:
            return '文本内容过少，无法生成推荐回复！请至少转发>3句话的聊天记录'
        for s in self.ctx[len(self.ctx)-3:]:
            last+=s
        prompts = f'现在你将扮演一个中文互联网的网民，会使用一些网络用于，正在使用微信聊天，善解人意。我现在在与一位MBTI属性为{type}的朋友聊天，以下是我们聊天的最后几句话:"{last}",请帮我想出一个比较好的回复'
        if self.require != '':
            prompts+=f',当前语境关键词为：{self.require}'
        state,res= gpt_35_api_stream([{'role': 'user','content': prompts},])
        if not state:
            return 'GPT API 调用失败，无法为您生成推荐回复!'
        return res
    
    def get_ctx(self):
        return self.ctx

    def get_trans(self):
        return self.trans


glo_usr: Dict[str,UserNode] ={}
admin_usr = []
use_recommend = True
if os.path.exists('./bot.dat'):
    admin_usr = torch.load('./bot.dat')

lock = Lock()

clear_flag = False
class MyBot(Wechaty):
    
    async def send_text(self,from_contact,text):
        conversation = from_contact
        await conversation.ready()
        await conversation.say(text)
    async def get_usr(self):
        while True:
           
            if len(glo_usr) == 0:
                await asyncio.sleep(0.1)
            for k,v in glo_usr:
                yield k,v
    async def on_ready(self, payload: EventReadyPayload) -> Coroutine[Any, Any, None]:
        asyncio.get_running_loop().create_task(process_loop(self))
         
    async def on_friendship(self, friendship: Friendship) -> Coroutine[Any, Any, None]:
        
        await friendship.accept()
        await friendship.contact().say('嗨～ 欢迎使用Chatify ! \nSTEP 1：向聊天框输入: /start \nSTEP 2: 待ChatiFy Bot回应后，找到想要了解好友的微信，多选聊天记录并逐条转发给ChatiFy, 或复制朋友圈等文案粘贴到聊天框\n STEP 3：发送完成后，向聊天框输入: /stop \nSTEP 4: 静候ChatiFy Bot的回复~')
    
    async def on_message(self, msg: Message):
        """
        listen for message event
        """
        global glo_usr
        from_contact: Optional[Contact] = msg.talker()
        to_contact: Optional[Contact] = msg.to()
        
        text = msg.text()
        type = msg.type()
        if not from_contact or not to_contact or type!=MessageType.MESSAGE_TYPE_TEXT:
            return
        room: Optional[Room] = msg.room()
        if to_contact.contact_id != 'filehelper' and to_contact.contact_id !=self.contact_id:
            return
        if text == 'ding':
            await self.send_text(from_contact,"dong")
        elif text.startswith('/admin'):
            parts = text.split(' ')
            if len(parts)<2:
                return
            if parts[1] == 'breaker2023':
                if from_contact.contact_id in admin_usr:
                    await self.send_text(from_contact,"当前用户已经是管理员了!")
                else:
                    admin_usr.append(from_contact.contact_id)
                    await self.send_text(from_contact,"[Admin]\n添加用户:"+from_contact.name+" 为管理员！")
        elif text.startswith('/info'):
            if from_contact.contact_id not in admin_usr:
                return
            parts = text.split(' ')
            if len(parts)<2:
                return
            if parts[1] == 'member':
                await self.send_text(from_contact,"当前会话中的用户: "+str(len(glo_usr))+"个")
            elif parts[1] == 'mem':
                mem = psutil.virtual_memory()
                # 系统总计内存
                zj = float(mem.total) / 1024 / 1024 / 1024
                # 系统已经使用内存
                ysy = float(mem.used) / 1024 / 1024 / 1024
                await self.send_text(from_contact,"当前系统内存占用:"+str(ysy/zj*100))
            elif parts[1] == 'disk':
                gb = 1024 ** 3 #GB == gigabyte 
                r = psutil.disk_usage('/') #查看磁盘的使用情况
                #print(r)
                # print('总的磁盘空间: {:6.2f} GB '.format( total_b / gb))
                # print('已经使用的 : {:6.2f} GB '.format( used_b / gb))
                # print('未使用的 : {:6.2f} GB '.format( free_b / gb))
                await self.send_text(from_contact,"当前系统硬盘占用:"+str(r.percent))
            elif parts[1] == 'save':
                torch.save(admin_usr,'./bot.dat')
                await self.send_text(from_contact,"保存当前管理员信息到硬盘，共"+str(len(admin_usr))+'个')
            elif parts[1] == 'gpt':
                global use_recommend
                use_recommend = not use_recommend
                await self.send_text(from_contact,"建议模式："+('开启'if use_recommend else '关闭')) 
            elif parts[1] == 'clear':
                lock.acquire()
                await self.send_text(from_contact,"获取锁成功！当前人数:"+str(len(glo_usr)))
                glo_usr={}
                lock.release() 
        elif text == '/start':
            if from_contact.contact_id in glo_usr:
                if glo_usr[from_contact.contact_id].get_state()!=10:
                    await self.send_text(from_contact,"Chatify:\n用户:"+from_contact.name+"\n[ 已经 ] 输入过/start了，请转发聊天记录或者输入 /stop 来停止")
                    return
            glo_usr[from_contact.contact_id] = UserNode()
            await self.send_text(from_contact,"Chatify:\n用户:"+from_contact.name+"\n文本分析开始!请转发聊天记录(逐条)到此聊天窗口")
        elif text.startswith('/stop'):
            parts = text.split('@')
            req = ''
            if len(parts)==2:
                req = parts[1]
            if from_contact.contact_id in glo_usr:
                if glo_usr[from_contact.contact_id].get_state()!=0:
                    await self.send_text(from_contact,"Chatify:\n文本输入已经停止，请耐心等待服务器分析数据...")
                else:
                    glo_usr[from_contact.contact_id].require = req
                    glo_usr[from_contact.contact_id].set_state(1)
                    await self.send_text(from_contact,"Chatify:\n用户:"+from_contact.name+"\n文本获取 停止，接下来将对聊天文本进行分析，请稍等...")
                    # state,out = await glo_usr[from_contact.contact_id].process()
                    # if not state:
                    #     await self.send_text(from_contact,"错误！翻译API调用失败!(空的聊天文本？)\n:(\n请尝试输入/start重新开始会话")
                    #     del glo_usr[from_contact.contact_id]
                    # else:
                        
                    #     # photo=FileBox.from_file('mbti/'+out+'.png')
                    #     # await from_contact.say(photo)
                    #     await self.send_text(from_contact,"属性:"+out)
                    # if use_recommend:
                    #     out = await glo_usr[from_contact.contact_id].recommend(out)
                    #     await self.send_text(from_contact,"推荐回复:\n"+out)
                    # del glo_usr[from_contact.contact_id]
            else:
                await self.send_text(from_contact,"Chatify:\n未获取到聊天文本信息！\n请输入/start开始")
        else:
            if from_contact.contact_id in glo_usr:
                if glo_usr[from_contact.contact_id].get_state()==0:
                    glo_usr[from_contact.contact_id].add_ctx(text)
s=None
async def process_loop(bot:MyBot):
    global glo_usr,s
    while True:
        await asyncio.sleep(1)
        #print('1',end='')
        for k,v in glo_usr.items():
            if v.get_state() == 2:
                glo_usr[k].set_state(10)
                # if v.get_mbti()!='':
                p = 'txt/'+cute_labels[v.mbti]+'.png'
                await bot.send_text(bot.Contact(k),v.get_output())
                if os.path.exists(p):
                    photo = FileBox.from_file(p)
                    await bot.Contact(k).say(photo)
            elif v.get_state() == 3:
                glo_usr[k].set_state(10)
                
                await bot.send_text(bot.Contact(k),'数据计算失败！')
            
            
                
os.environ['TOKEN'] = "1fe5f846-3cfb-401d-b20c-XXXXX"
os.environ['WECHATY_PUPPET_SERVICE_ENDPOINT'] = "127.0.0.1:9001"
bot = MyBot()
def test():
    global s,dead,lock,glo_usr,bot,lock
    while True:
        time.sleep(0.1)
        lock.acquire()
        for k,v in glo_usr.items():
            if v.get_state() == 1:
                state,out = v.process()
                if not state:
                    glo_usr[k].set_state(3)
                    continue
                else:
                    
                    # photo=FileBox.from_file('mbti/'+out+'.png')
                    # await from_contact.say(photo)
                    rec = v.recommend(out)
                    
                    glo_usr[k].set_output('TA的属性为：'+cute_labels[out]+'('+out+')\n推荐回复 '+rec+'')
                    glo_usr[k].set_mbti(out)
                    glo_usr[k].set_state(2)
        lock.release()

t0 = Thread(target= test,name='process_loop',daemon=True)
t0.start()
tasks = [
    asyncio.ensure_future(bot.start()),
    asyncio.ensure_future(process_loop(bot))
    ]
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))
#asyncio.run(asyncio.wait(tasks))


from moviepy.editor import *
import os
import joblib as job
import wave
from pysubs2 import SSAFile, SSAEvent, make_time
from pydub import AudioSegment
from pydub.silence import split_on_silence

from pdb import set_trace as stop



class Tv_extration:
    def __init__(self):
        self.read_base = '../../'
        self.save_base = '../../pre_data'
        self.Tv_friends = 'friends'
        self.save_utt_audio_pairs_path=''
        self.audio_path = ''
        self.subtitle_time_pairs = None # for all pair info
        self.seasons_length = [24,24,25,24,24,25,24,24,23,17]
        # self.seasons_length = [2]
        self.Tv_audios = {}
        self.Tv_SubTimePairs = {}
        self.UttAudiopairs = []
    


    def get_audio_from_Tv(self,): # get the audio from video using and output wav file
        for sea, lent in enumerate(self.seasons_length):
            drama_audio_dict = {}
            for epi in range(lent):
                drama_audio_dict[str(epi+1).zfill(2)]= self.get_audio_from_drama(sea+1,epi+1)
                # self.get_audio_from_drama(sea+1,epi+1)
            self.Tv_audios[str(sea+1).zfill(2)] = drama_audio_dict

    def get_audio_from_drama(self,season,episode ): # get drama wav audio file
        read_path = os.path.join(self.read_base,'S'+str(season).zfill(2),'friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2)+'.mp4')
        save_path = os.path.join(self.save_base,self.Tv_friends,'S'+str(season).zfill(2),'audio', 'friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2) +'.wav')
        if not os.path.exists(os.path.join(self.save_base,self.Tv_friends,'S'+str(season).zfill(2),'audio')):
            os.makedirs(os.path.join(self.save_base,self.Tv_friends,'S'+str(season).zfill(2),'audio'))
        if os.path.exists(save_path):
            # return AudioSegment.from_wav(save_path)
            return save_path
        else:
            video = VideoFileClip(read_path)
            audio = video.audio
            audio.write_audiofile(save_path)
            # return  AudioSegment.from_wav(save_path)
            return  save_path

    def get_utteraces_from_subtile(self,season,episode): # extract the subtitle time fragment from subtitle file
        read_path = os.path.join(self.read_base,'subtitle','season'+str(season),'applied','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2)+'.srt')
        subs = SSAFile.load(read_path,encoding='ISO-8859-1',format='srt')
        return subs

    def get_utteraces_from_Tv(self,): # extract all the time fragment from  subtitles
        for sea,lent in enumerate(self.seasons_length):
            drama_sub_dict = {}
            for epi in range(lent):
                drama_sub_dict[str(epi+1).zfill(2)] = self.get_utteraces_from_subtile(sea+1,epi+1)
            self.Tv_SubTimePairs[str(sea+1).zfill(2)] = drama_sub_dict

    def get_Tv_video_seg(self,):
        for sea,lent in enumerate(self.seasons_length):
            for epi in range(lent):
                print("Tv video seg (season,episode):({},{})".format(sea+1,epi+1))
                self.get_video_seg(sea+1,epi+1)
            
    def get_video_seg(self,season,episode):
        SubTimepairs = self.Tv_SubTimePairs[str(season).zfill(2)][str(episode).zfill(2)]
        original_file = os.path.join(self.read_base,'S'+str(season).zfill(2),'friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2)+'.mp4')
        # if original_file.endswith('friends.s01e03.mp4'):
        #     stop()
        videoSource = VideoFileClip(original_file)
        for i,sub in enumerate(SubTimepairs):
            
            new_file = os.path.join(self.save_base,'friendsVideoSeg','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2),'seg'+str(i+1).zfill(3)+ '.mp4')
            
            if not os.path.exists(os.path.join(self.save_base,'friendsVideoSeg','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2))):
                os.makedirs(os.path.join(self.save_base,'friendsVideoSeg','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2)))
            if not os.path.exists(new_file):
                try:
                    video = videoSource.subclip(sub.start/1000,sub.end/1000+0.3)
                    video.write_videofile(new_file)

                    
                except:
                    print("season:{},eposode:{},sub:{} is  throwed".format(season,episode,i))

    def get_TV_UttTimePairs(self,): # extract the utt-audio pairs based on the time-subtitle pairs 
        for sea,lent in enumerate(self.seasons_length):
            season_UttAudiopair = []
            for epi in range(lent):
                print("Tv audio seg (season,episode):({},{})".format(sea+1,epi+1))
                season_UttAudiopair.append(self.get_drama_UttTimePairs(sea+1,epi+1))
            self.UttAudiopairs.append(season_UttAudiopair)
        
        job.dump(self.UttAudiopairs,os.path.join(self.save_base,'UttAudioPairs.job'))
        
    def get_drama_UttTimePairs(self,season,episode):
        AudioSound = AudioSegment.from_wav(self.Tv_audios[str(season).zfill(2)][str(episode).zfill(2)])
        SubTimepairs = self.Tv_SubTimePairs[str(season).zfill(2)][str(episode).zfill(2)]
        drama_UttAudiopair = []
        for i,sub in enumerate(SubTimepairs):
            buff = AudioSound[sub.start: sub.end+300]  # 字符串索引切割
            if not os.path.exists(os.path.join(self.save_base,'friendsAudioSeg','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2))):
                os.makedirs(os.path.join(self.save_base,'friendsAudioSeg','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2)))
            buff.export(os.path.join(self.save_base,'friendsAudioSeg','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2),'seg'+str(i+1).zfill(3)+ '.wav'), format='wav')     
            drama_UttAudiopair.append(UttAudioPair(sub,season,episode,i+1))
        return drama_UttAudiopair

    def save_to_joblib(self,obj,file_path):
        job.dump(obj,file_path)

    def run(self,):  # the main preprocess
        self.get_audio_from_Tv() # video to wav audio
        self.get_utteraces_from_Tv() # get time subtitle pairs 
        print('='*10,'sucess audio utterance','='*10)
        utt_video_pairs = self.get_Tv_video_seg() # get utt-audio pairs
        utt_audio_pairs = self.get_TV_UttTimePairs() # get utt-audio pairs
        

class UttAudioPair:
    def __init__(self,subFile,season,episode,segId):
        self.subFile = subFile
        self.season = season
        self.episode = episode
        self.segId = segId

if __name__ == '__main__' :
    Tv_extration().run()













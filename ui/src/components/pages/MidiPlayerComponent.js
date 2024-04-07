import React from 'react';
import MidiPlayer from 'react-midi-player';
import './css/MidiPlayerComponent.scss';

const MidiPlayerComponent = () => {
    return (
      <div className="container">
        <div className='midi-container'>
          <h2 style={{color:'#67584d'}}>We composed it for you, now it's time to listen!</h2>
          <h2 style={{color:'#67584d'}}>Enjoy your own music...</h2>
          <div className="midi-player">
            <MidiPlayer
              src="/composed_music.mid"
              onPlay={(e) => console.log('Play', e)}
              onStop={(e) => console.log('Stop', e)}
              onPause={(e) => console.log('Pause', e)}
              width={250}
            />
          </div>
        </div>
      
      </div>
    );
  };
  
  export default MidiPlayerComponent;
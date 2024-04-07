import React from 'react';
import ReactLoading from 'react-loading';
import './css/Loading.scss';

const LoadingScreen = () => {
    return (
        <div className="loading">
            <div>
                <h1 style={{color:'#1b5074',fontSize:24}}>We are preparing a magnificent music for you</h1>
                <h1 style={{color:'#1b5074',fontSize:24}}>Please wait..</h1>
            </div>
            <div style={{margin: 'auto'}}>
                <ReactLoading type="spinningBubbles" color="#1b5074" height={100} width={100}/>
            </div>
        </div>
    );
};

export default LoadingScreen;

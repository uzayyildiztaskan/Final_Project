import React, { useState } from "react";
import './css/Home.scss';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Homepage() {
    const [userInput, setUserInput] = useState('');
    let navigate = useNavigate();

    const handleSubmit = async () => {
        try {
            navigate('/midi');
            const response = await axios.post('http://localhost:5000/api/ai-function', { text: userInput });
            console.log(response.data); 
        
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <section id="main">
            <div className="header">
                <h1 className="navbar-brand">VIRTUAICOMPOSER</h1>
            </div>
            <div className="prompt-container">
                <h2 style={{color:'#67584d'}}>Ready to create music?</h2>
                <p style={{color:'#67584d'}}>Tell us what you have in mind, and we'll help you compose it!</p>
                <textarea 
                  placeholder="Please enter what kind of music you want ..."
                  value={userInput}
                  onChange={e => setUserInput(e.target.value)}
                />
                <button onClick={handleSubmit} className="btn" style={{ textDecoration: "none", color:"#003c4c" }}>
                    Compose Now
                </button>
            </div>
        </section>
    );
}


export default Homepage;

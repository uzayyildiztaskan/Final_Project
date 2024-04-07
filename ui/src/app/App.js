import './App.css';
import { ThemeContext, themes } from "../api/Theme";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Welcome from "../components/pages/Welcome";
import Home from "../components/pages/Home";
import MidiPlayer from "../components/pages/MidiPlayerComponent";

function App() {
    return (
        <ThemeContext.Provider value={themes.light}>
            <>
                <Router>
                    <Routes>
                        <Route exact path="/" element={<Welcome />} />
                        <Route exact path="/home" element={<Home />} />
                        <Route exact path="/midi" element={<MidiPlayer />} />

                    </Routes>
                </Router>
            </>
        </ThemeContext.Provider>
    );
}

export default App;
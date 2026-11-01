import '@fontsource/ibm-plex-sans/latin-400.css';
import '@fontsource/ibm-plex-sans/latin-500.css';
import '@fontsource/ibm-plex-sans/latin-600.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import { createRoot } from 'react-dom/client';

import { App } from './App.tsx';
import './carbon.scss';
import './styles.css';

const root = document.getElementById('root');
if (root) createRoot(root).render(<App />);

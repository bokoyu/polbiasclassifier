import React from 'react';
import { Flex, Box, Link } from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

function NavBar() {
  return (
    <Flex bg="gray.100" p={4} mb={4} justify="flex-start" align="center">
      <Box mr={6}>
        <Link as={RouterLink} to="/">Home</Link>
      </Box>
      <Box mr={6}>
        <Link as={RouterLink} to="/evaluate">Evaluate</Link>
      </Box>
      <Box mr={6}>
        <Link as={RouterLink} to="/train">Train</Link>
      </Box>
    </Flex>
  );
}

export default NavBar;
